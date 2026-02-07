"""Baseline methods for ACM MM comparison experiments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import time
import logging

from major_tom.llm.base import LLMBackend

logger = logging.getLogger(__name__)


@dataclass
class MethodStats:
    """Statistics collected during method execution."""

    llm_calls: int = 0
    vlm_calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    semantic_filtered: int = 0
    event_count: int = 0
    summary_count: int = 0
    latencies: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "llm_calls": self.llm_calls,
            "vlm_calls": self.vlm_calls,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_hits": self.cache_hits,
            "semantic_filtered": self.semantic_filtered,
            "event_count": self.event_count,
            "summary_count": self.summary_count,
            "latencies": self.latencies,
        }


class BaselineMethod(ABC):
    """Abstract base class for baseline methods."""

    name: str = "base"

    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        self.llm = llm_backend
        self.stats = MethodStats()

    @abstractmethod
    def process_event(self, event: Dict) -> Tuple[str, Optional[str]]:
        """
        Process a single activity event.

        Args:
            event: Dictionary containing event data (app, title, duration, etc.)

        Returns:
            Tuple of (decision, summary)
            - decision: "SNAPSHOT" or "SKIP"
            - summary: Generated summary text or None
        """
        pass

    def reset_stats(self):
        """Reset statistics for new experiment run."""
        self.stats = MethodStats()

    def _build_summarization_prompt(self, event: Dict) -> str:
        """Build prompt for activity summarization."""
        return f"""Summarize this activity briefly in one sentence:
Application: {event.get('app', 'Unknown')}
Window Title: {event.get('title', 'Unknown')}
Duration: {event.get('duration_seconds', 0):.0f} seconds

Summary:"""


class NaiveFullBaseline(BaselineMethod):
    """
    B0: Naive Full Processing
    Process every event with full LLM + VLM (if available).
    This serves as the upper bound for quality and cost.
    """

    name = "naive_full"

    def process_event(self, event: Dict) -> Tuple[str, Optional[str]]:
        start = time.perf_counter()
        self.stats.event_count += 1

        if not self.llm:
            # No LLM backend - generate simple summary
            summary = f"Activity in {event.get('app', 'Unknown')}: {event.get('title', '')[:50]}"
            self.stats.latencies.append((time.perf_counter() - start) * 1000)
            self.stats.summary_count += 1
            return "SNAPSHOT", summary

        # Always call LLM for summarization
        prompt = self._build_summarization_prompt(event)
        response = self.llm.generate("brain", prompt)

        self.stats.llm_calls += 1
        self.stats.total_tokens += response.total_tokens
        self.stats.prompt_tokens += response.prompt_tokens
        self.stats.completion_tokens += response.completion_tokens

        # Call VLM if screenshot available
        if event.get("screenshot_path"):
            try:
                vlm_prompt = "Describe what you see on this screen in one sentence."
                vlm_response = self.llm.generate("eye", vlm_prompt)
                self.stats.vlm_calls += 1
                self.stats.total_tokens += vlm_response.total_tokens
            except Exception as e:
                logger.debug(f"VLM call failed: {e}")

        latency = (time.perf_counter() - start) * 1000
        self.stats.latencies.append(latency)
        self.stats.summary_count += 1

        return "SNAPSHOT", response.text.strip()


class RuleBasedBaseline(BaselineMethod):
    """
    B1: Rule-Based Filtering
    Simple rules based on application name and keywords.
    Fast but low quality - serves as lower bound.
    """

    name = "rule_based"

    # Apps to always skip
    SKIP_APPS: Set[str] = {
        "Finder", "System Preferences", "Activity Monitor",
        "System Information", "Disk Utility", "Console",
        "Keychain Access", "Migration Assistant",
    }

    # Apps that are always important
    IMPORTANT_APPS: Set[str] = {
        "VS Code", "Xcode", "PyCharm", "IntelliJ",
        "Chrome", "Safari", "Firefox",
        "Slack", "Discord", "Zoom",
        "Notes", "Obsidian", "Notion",
        "Word", "Pages", "Docs",
    }

    # Keywords in titles that indicate importance
    IMPORTANT_KEYWORDS: Set[str] = {
        "meeting", "call", "project", "deadline",
        "report", "presentation", "review",
        ".py", ".js", ".ts", ".java", ".swift",
        "github", "gitlab", "stackoverflow",
    }

    def process_event(self, event: Dict) -> Tuple[str, Optional[str]]:
        start = time.perf_counter()
        self.stats.event_count += 1

        app = event.get("app", "")
        title = event.get("title", "").lower()

        # Rule 1: Skip system apps
        if app in self.SKIP_APPS:
            self.stats.latencies.append((time.perf_counter() - start) * 1000)
            return "SKIP", None

        # Rule 2: Important apps always snapshot
        if app in self.IMPORTANT_APPS:
            summary = f"Working in {app}: {event.get('title', '')[:50]}"
            self.stats.latencies.append((time.perf_counter() - start) * 1000)
            self.stats.summary_count += 1
            return "SNAPSHOT", summary

        # Rule 3: Check for important keywords
        for keyword in self.IMPORTANT_KEYWORDS:
            if keyword in title:
                summary = f"Activity related to '{keyword}' in {app}"
                self.stats.latencies.append((time.perf_counter() - start) * 1000)
                self.stats.summary_count += 1
                return "SNAPSHOT", summary

        # Rule 4: Default skip
        self.stats.latencies.append((time.perf_counter() - start) * 1000)
        return "SKIP", None


class FixedSamplingBaseline(BaselineMethod):
    """
    B2: Fixed Interval Sampling
    Sample at fixed time intervals regardless of content.
    Simple but misses important short activities.
    """

    name = "fixed_sampling"

    def __init__(
        self,
        llm_backend: Optional[LLMBackend] = None,
        interval_seconds: int = 300,
    ):
        super().__init__(llm_backend)
        self.interval = interval_seconds
        self.last_sample_time = 0.0

    def process_event(self, event: Dict) -> Tuple[str, Optional[str]]:
        start = time.perf_counter()
        self.stats.event_count += 1

        current_time = event.get("timestamp", time.time())

        # Check if it's time to sample
        if current_time - self.last_sample_time < self.interval:
            self.stats.latencies.append((time.perf_counter() - start) * 1000)
            return "SKIP", None

        # Time to sample
        self.last_sample_time = current_time

        if not self.llm:
            summary = f"[{current_time}] {event.get('app', 'Unknown')}: {event.get('title', '')[:50]}"
            self.stats.latencies.append((time.perf_counter() - start) * 1000)
            self.stats.summary_count += 1
            return "SNAPSHOT", summary

        # Generate summary with LLM
        prompt = self._build_summarization_prompt(event)
        response = self.llm.generate("brain", prompt)

        self.stats.llm_calls += 1
        self.stats.total_tokens += response.total_tokens
        self.stats.prompt_tokens += response.prompt_tokens
        self.stats.completion_tokens += response.completion_tokens

        latency = (time.perf_counter() - start) * 1000
        self.stats.latencies.append(latency)
        self.stats.summary_count += 1

        return "SNAPSHOT", response.text.strip()

    def reset_stats(self):
        super().reset_stats()
        self.last_sample_time = 0.0


class EmbeddingOnlyBaseline(BaselineMethod):
    """
    B3: Embedding-Only Filtering
    Use embedding similarity to filter redundant events.
    No LLM calls - only embedding model.
    """

    name = "embedding_only"

    def __init__(
        self,
        llm_backend: Optional[LLMBackend] = None,
        similarity_threshold: float = 0.85,
        history_size: int = 10,
    ):
        super().__init__(llm_backend)
        self.threshold = similarity_threshold
        self.max_history = history_size
        self.recent_embeddings: List[List[float]] = []
        self.recent_texts: List[str] = []

    def process_event(self, event: Dict) -> Tuple[str, Optional[str]]:
        start = time.perf_counter()
        self.stats.event_count += 1

        # Build text representation
        text = f"{event.get('app', '')} {event.get('title', '')}"

        if not self.llm:
            # Without LLM, use simple text similarity
            is_similar = any(
                self._text_similarity(text, prev) > self.threshold
                for prev in self.recent_texts
            )

            if is_similar:
                self.stats.semantic_filtered += 1
                self.stats.latencies.append((time.perf_counter() - start) * 1000)
                return "SKIP", None

            # Update history
            self.recent_texts.append(text)
            if len(self.recent_texts) > self.max_history:
                self.recent_texts.pop(0)

            summary = f"Activity in {event.get('app', 'Unknown')}: {event.get('title', '')[:100]}"
            self.stats.latencies.append((time.perf_counter() - start) * 1000)
            self.stats.summary_count += 1
            return "SNAPSHOT", summary

        # Get embedding for current event
        embed_response = self.llm.embed("embedding", text)
        current_embedding = embed_response.vector

        self.stats.total_tokens += embed_response.prompt_tokens

        # Check similarity with recent events
        is_similar = False
        for prev_emb in self.recent_embeddings:
            similarity = self._cosine_similarity(current_embedding, prev_emb)
            if similarity > self.threshold:
                is_similar = True
                self.stats.semantic_filtered += 1
                break

        # Update history
        self.recent_embeddings.append(current_embedding)
        if len(self.recent_embeddings) > self.max_history:
            self.recent_embeddings.pop(0)

        latency = (time.perf_counter() - start) * 1000
        self.stats.latencies.append(latency)

        if is_similar:
            return "SKIP", None

        # Generate simple summary (no LLM)
        summary = f"Activity in {event.get('app', 'Unknown')}: {event.get('title', '')[:100]}"
        self.stats.summary_count += 1
        return "SNAPSHOT", summary

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _text_similarity(self, a: str, b: str) -> float:
        """Simple text similarity using Jaccard index."""
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)

        return intersection / union if union > 0 else 0.0

    def reset_stats(self):
        super().reset_stats()
        self.recent_embeddings = []
        self.recent_texts = []


class LLMClassifierBaseline(BaselineMethod):
    """
    B4: LLM as Binary Classifier
    Use LLM to decide whether to record, then generate summary.
    Two LLM calls per snapshot: classify + summarize.
    """

    name = "llm_classifier"

    CLASSIFIER_PROMPT = """Determine if this activity is worth recording in a personal journal.

Application: {app}
Window Title: {title}
Duration: {duration} seconds
Typing Speed: {kpm} keystrokes/min
Click Rate: {cpm} clicks/min

Consider:
- Is this work or productive activity?
- Does it represent meaningful progress?
- Would the user want to remember this later?

Reply with only YES or NO."""

    def process_event(self, event: Dict) -> Tuple[str, Optional[str]]:
        start = time.perf_counter()
        self.stats.event_count += 1

        if not self.llm:
            # Fallback: use simple heuristics
            summary = f"Activity in {event.get('app', 'Unknown')}: {event.get('title', '')[:50]}"
            self.stats.latencies.append((time.perf_counter() - start) * 1000)
            self.stats.summary_count += 1
            return "SNAPSHOT", summary

        # Step 1: Classify with LLM
        classify_prompt = self.CLASSIFIER_PROMPT.format(
            app=event.get("app", "Unknown"),
            title=event.get("title", "Unknown"),
            duration=event.get("duration_seconds", 0),
            kpm=event.get("kpm", 0),
            cpm=event.get("cpm", 0),
        )

        classify_response = self.llm.generate("brain", classify_prompt, max_tokens=10)

        self.stats.llm_calls += 1
        self.stats.total_tokens += classify_response.total_tokens
        self.stats.prompt_tokens += classify_response.prompt_tokens
        self.stats.completion_tokens += classify_response.completion_tokens

        # Parse decision
        decision_text = classify_response.text.strip().upper()
        should_record = "YES" in decision_text

        if not should_record:
            self.stats.latencies.append((time.perf_counter() - start) * 1000)
            return "SKIP", None

        # Step 2: Generate summary with LLM
        summary_prompt = self._build_summarization_prompt(event)
        summary_response = self.llm.generate("brain", summary_prompt)

        self.stats.llm_calls += 1
        self.stats.total_tokens += summary_response.total_tokens
        self.stats.prompt_tokens += summary_response.prompt_tokens
        self.stats.completion_tokens += summary_response.completion_tokens

        latency = (time.perf_counter() - start) * 1000
        self.stats.latencies.append(latency)
        self.stats.summary_count += 1

        return "SNAPSHOT", summary_response.text.strip()


class HSRMethod(BaselineMethod):
    """
    Our Method: Hierarchical Semantic Routing (HSR)
    Three-layer adaptive routing for efficient activity summarization.

    Layer 1: Semantic Gating (embedding similarity)
    Layer 2: Decision Cache (context-aware caching)
    Layer 3: LLM Analysis (fine-grained processing)
    """

    name = "hsr"

    def __init__(
        self,
        llm_backend: Optional[LLMBackend] = None,
        semantic_threshold: float = 0.30,
        cache_ttl: int = 300,
        history_size: int = 10,
    ):
        super().__init__(llm_backend)

        # Layer 1: Semantic Gating
        self.semantic_threshold = semantic_threshold
        self.history_size = history_size
        self.recent_embeddings: List[List[float]] = []

        # Layer 2: Decision Cache
        self.decision_cache: Dict[str, Tuple[str, float]] = {}  # key -> (decision, timestamp)
        self.cache_ttl = cache_ttl

    def process_event(self, event: Dict) -> Tuple[str, Optional[str]]:
        start = time.perf_counter()
        self.stats.event_count += 1
        current_time = event.get("timestamp", time.time())

        app = event.get("app", "")
        title = event.get("title", "")
        text = f"{app} {title}"

        # ===== Layer 1: Semantic Gating =====
        if self.llm:
            embed_response = self.llm.embed("embedding", text)
            current_embedding = embed_response.vector
            self.stats.total_tokens += embed_response.prompt_tokens

            # Check if semantically similar to recent events
            max_similarity = 0.0
            for prev_emb in self.recent_embeddings:
                similarity = self._cosine_similarity(current_embedding, prev_emb)
                max_similarity = max(max_similarity, similarity)

            # Update embedding history
            self.recent_embeddings.append(current_embedding)
            if len(self.recent_embeddings) > self.history_size:
                self.recent_embeddings.pop(0)

            # If highly similar, skip (semantic gating)
            if max_similarity > (1 - self.semantic_threshold):
                self.stats.semantic_filtered += 1
                self.stats.latencies.append((time.perf_counter() - start) * 1000)
                return "SKIP", None

        # ===== Layer 2: Decision Cache =====
        cache_key = self._get_cache_key(event)

        if cache_key in self.decision_cache:
            cached_decision, cached_time = self.decision_cache[cache_key]
            if current_time - cached_time < self.cache_ttl:
                self.stats.cache_hits += 1
                self.stats.latencies.append((time.perf_counter() - start) * 1000)

                if cached_decision == "SKIP":
                    return "SKIP", None
                else:
                    # Generate quick summary without full LLM
                    summary = f"Continued work in {app}: {title[:50]}"
                    self.stats.summary_count += 1
                    return "SNAPSHOT", summary

        # ===== Layer 3: LLM Analysis =====
        if not self.llm:
            summary = f"Activity in {app}: {title[:100]}"
            self.stats.latencies.append((time.perf_counter() - start) * 1000)
            self.stats.summary_count += 1
            return "SNAPSHOT", summary

        # Full LLM analysis for novel/important events
        prompt = self._build_analysis_prompt(event)
        response = self.llm.generate("brain", prompt)

        self.stats.llm_calls += 1
        self.stats.total_tokens += response.total_tokens
        self.stats.prompt_tokens += response.prompt_tokens
        self.stats.completion_tokens += response.completion_tokens

        # Parse response and cache decision
        decision, summary = self._parse_response(response.text, event)

        self.decision_cache[cache_key] = (decision, current_time)

        latency = (time.perf_counter() - start) * 1000
        self.stats.latencies.append(latency)

        if decision == "SNAPSHOT":
            self.stats.summary_count += 1

        return decision, summary

    def _get_cache_key(self, event: Dict) -> str:
        """Generate cache key from event context."""
        app = event.get("app", "")
        # Normalize title to reduce cache misses
        title = event.get("title", "")
        # Remove variable parts (line numbers, timestamps, etc.)
        import re
        normalized_title = re.sub(r'\d+', '#', title)
        normalized_title = re.sub(r'[:\-]\s*\d+', '', normalized_title)

        return f"{app}|{normalized_title[:50]}"

    def _build_analysis_prompt(self, event: Dict) -> str:
        """Build prompt for LLM analysis."""
        return f"""Analyze this activity and decide if it should be recorded:

Application: {event.get('app', 'Unknown')}
Window Title: {event.get('title', 'Unknown')}
Duration: {event.get('duration_seconds', 0):.0f} seconds
Typing: {event.get('kpm', 0):.0f} keys/min
Clicks: {event.get('cpm', 0):.0f} clicks/min

Reply in this format:
DECISION: SNAPSHOT or SKIP
SUMMARY: <one sentence summary if SNAPSHOT, empty if SKIP>"""

    def _parse_response(self, response: str, event: Dict) -> Tuple[str, Optional[str]]:
        """Parse LLM response to extract decision and summary."""
        response_upper = response.upper()

        if "SKIP" in response_upper and "SNAPSHOT" not in response_upper:
            return "SKIP", None

        # Extract summary
        summary = None
        if "SUMMARY:" in response:
            summary = response.split("SUMMARY:")[-1].strip()
        elif "SUMMARY：" in response:  # Chinese colon
            summary = response.split("SUMMARY：")[-1].strip()
        else:
            # Fallback: use the whole response or generate simple summary
            summary = response.strip() if len(response.strip()) < 200 else \
                f"Working in {event.get('app', 'Unknown')}: {event.get('title', '')[:50]}"

        return "SNAPSHOT", summary

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def reset_stats(self):
        super().reset_stats()
        self.recent_embeddings = []
        self.decision_cache = {}


# Method registry
BASELINE_METHODS = {
    "naive_full": NaiveFullBaseline,
    "rule_based": RuleBasedBaseline,
    "fixed_sampling": FixedSamplingBaseline,
    "embedding_only": EmbeddingOnlyBaseline,
    "llm_classifier": LLMClassifierBaseline,
    "hsr": HSRMethod,
}


def get_method(name: str, llm_backend: Optional[LLMBackend] = None, **kwargs) -> BaselineMethod:
    """Get a baseline method by name."""
    if name not in BASELINE_METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(BASELINE_METHODS.keys())}")

    return BASELINE_METHODS[name](llm_backend=llm_backend, **kwargs)
