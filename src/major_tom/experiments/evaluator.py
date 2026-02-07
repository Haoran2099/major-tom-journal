"""Automated evaluation metrics for journal quality and memory effectiveness."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from major_tom.llm.base import LLMBackend

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Aggregated evaluation results across experiment runs."""

    dimension: str = ""
    condition: str = ""
    runs: int = 0

    # Journal quality
    relevance_scores: List[float] = field(default_factory=list)
    redundancy_rates: List[float] = field(default_factory=list)
    coverage_rates: List[float] = field(default_factory=list)
    information_densities: List[float] = field(default_factory=list)

    # Memory effectiveness
    context_pollution_rates: List[float] = field(default_factory=list)
    context_recall_accuracies: List[float] = field(default_factory=list)
    memory_isolation_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
            arr = np.array(values)
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
            }

        return {
            "dimension": self.dimension,
            "condition": self.condition,
            "runs": self.runs,
            "journal_quality": {
                "relevance_score": _stats(self.relevance_scores),
                "redundancy_rate": _stats(self.redundancy_rates),
                "coverage_rate": _stats(self.coverage_rates),
                "information_density": _stats(self.information_densities),
            },
            "memory_effectiveness": {
                "context_pollution_rate": _stats(self.context_pollution_rates),
                "context_recall_accuracy": _stats(self.context_recall_accuracies),
                "memory_isolation_score": _stats(self.memory_isolation_scores),
            },
        }


class JournalQualityEvaluator:
    """Automated evaluation of journal entries against ground truth."""

    def __init__(
        self,
        ground_truth_path: Optional[Path] = None,
        llm_backend: Optional[LLMBackend] = None,
    ):
        self._ground_truth = self._load_ground_truth(ground_truth_path) if ground_truth_path else []
        self._llm = llm_backend

    def _load_ground_truth(self, path: Path) -> List[Dict]:
        """Load ground truth annotations from JSONL."""
        entries = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        logger.info("Loaded %d ground truth annotations", len(entries))
        return entries

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text."""
        if self._llm is None:
            return None
        try:
            resp = self._llm.embed("qwen3-embedding:8b", text)
            if resp.vector:
                return np.array(resp.vector, dtype=np.float32)
        except Exception:
            pass
        return None

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def compute_relevance_score(self, entries: List[str]) -> float:
        """RS: Cosine similarity between journal entries and ground truth descriptions."""
        if not self._ground_truth or not entries or self._llm is None:
            return 0.0

        scores = []
        gt_texts = [g["description"] for g in self._ground_truth if g.get("description")]

        for entry in entries[:50]:  # Sample
            entry_vec = self._get_embedding(entry)
            if entry_vec is None:
                continue
            best = 0.0
            for gt in gt_texts:
                gt_vec = self._get_embedding(gt)
                if gt_vec is not None:
                    sim = self._cosine_sim(entry_vec, gt_vec)
                    best = max(best, sim)
            scores.append(best)

        return float(np.mean(scores)) if scores else 0.0

    def compute_redundancy_rate(self, entries: List[str]) -> float:
        """RR: Average pairwise similarity between adjacent entries (lower is better)."""
        if len(entries) < 2 or self._llm is None:
            return 0.0

        sims = []
        prev_vec = None
        for entry in entries:
            vec = self._get_embedding(entry)
            if vec is not None and prev_vec is not None:
                sims.append(self._cosine_sim(prev_vec, vec))
            prev_vec = vec

        return float(np.mean(sims)) if sims else 0.0

    def compute_coverage_rate(self, entries: List[str]) -> float:
        """ACR: Fraction of ground truth activities covered by journal entries."""
        if not self._ground_truth or not entries or self._llm is None:
            return 0.0

        gt_texts = [g["description"] for g in self._ground_truth if g.get("description")]
        covered = 0

        for gt in gt_texts:
            gt_vec = self._get_embedding(gt)
            if gt_vec is None:
                continue
            for entry in entries:
                entry_vec = self._get_embedding(entry)
                if entry_vec is not None:
                    if self._cosine_sim(gt_vec, entry_vec) > 0.5:
                        covered += 1
                        break

        return covered / len(gt_texts) if gt_texts else 0.0

    def compute_information_density(self, entries: List[str]) -> float:
        """ID: Average unique words per entry (proxy for concept density)."""
        if not entries:
            return 0.0
        densities = []
        for entry in entries:
            words = entry.split()
            unique = len(set(w.lower() for w in words))
            total = max(len(words), 1)
            densities.append(unique / total)
        return float(np.mean(densities))

    def compute_memory_isolation_score(self, memory_files: Dict[str, str]) -> float:
        """MIS: 1 - mean pairwise similarity between memory files."""
        if len(memory_files) < 2 or self._llm is None:
            return 1.0

        vectors = []
        for content in memory_files.values():
            vec = self._get_embedding(content[:2000])
            if vec is not None:
                vectors.append(vec)

        if len(vectors) < 2:
            return 1.0

        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(self._cosine_sim(vectors[i], vectors[j]))

        return 1.0 - float(np.mean(sims)) if sims else 1.0

    def evaluate_run(
        self,
        journal_entries: List[str],
        memory_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Evaluate a single experiment run."""
        return {
            "relevance_score": self.compute_relevance_score(journal_entries),
            "redundancy_rate": self.compute_redundancy_rate(journal_entries),
            "coverage_rate": self.compute_coverage_rate(journal_entries),
            "information_density": self.compute_information_density(journal_entries),
            "memory_isolation_score": self.compute_memory_isolation_score(memory_files or {}),
        }
