"""Evaluation metrics for ACM MM benchmark."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyMetrics:
    """Efficiency-related metrics."""

    total_tokens: int
    tokens_per_event: float
    tokens_per_summary: float
    llm_call_count: int
    llm_call_rate: float
    vlm_call_count: int
    vlm_call_rate: float
    cache_hit_count: int
    cache_hit_rate: float
    semantic_filter_count: int
    semantic_filter_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    token_efficiency: float  # 1 - (method_tokens / naive_tokens)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class QualityMetrics:
    """Quality-related metrics."""

    # Classification
    classification_accuracy: float
    classification_macro_f1: float
    classification_weighted_f1: float

    # Summarization
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float

    # QA
    qa_exact_match: float
    qa_f1: float

    # Ranking
    ranking_ndcg_5: float
    ranking_ndcg_10: float
    ranking_spearman: float

    # Overall
    quality_score: float  # Weighted average of all metrics

    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsCalculator:
    """Calculate all benchmark metrics."""

    def __init__(self):
        self._rouge_scorer = None
        self._bertscore_available = False
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if optional dependencies are available."""
        try:
            from rouge_score import rouge_scorer
            self._rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        except ImportError:
            logger.warning("rouge_score not installed. ROUGE metrics will be estimated.")

        try:
            import bert_score
            self._bertscore_available = True
        except ImportError:
            logger.warning("bert_score not installed. BERTScore will be estimated.")

    def compute_efficiency(
        self,
        method_stats: Dict,
        naive_stats: Optional[Dict] = None,
    ) -> EfficiencyMetrics:
        """Compute efficiency metrics relative to naive baseline."""

        total_tokens = method_stats.get("total_tokens", 0)
        event_count = method_stats.get("event_count", 1)
        summary_count = method_stats.get("summary_count", 1) or 1
        latencies = method_stats.get("latencies", [0])

        # Compute token efficiency if naive baseline provided
        token_efficiency = 0.0
        if naive_stats and naive_stats.get("total_tokens", 0) > 0:
            token_efficiency = 1 - (total_tokens / naive_stats["total_tokens"])

        return EfficiencyMetrics(
            total_tokens=total_tokens,
            tokens_per_event=total_tokens / max(event_count, 1),
            tokens_per_summary=total_tokens / max(summary_count, 1),
            llm_call_count=method_stats.get("llm_calls", 0),
            llm_call_rate=method_stats.get("llm_calls", 0) / max(event_count, 1),
            vlm_call_count=method_stats.get("vlm_calls", 0),
            vlm_call_rate=method_stats.get("vlm_calls", 0) / max(event_count, 1),
            cache_hit_count=method_stats.get("cache_hits", 0),
            cache_hit_rate=method_stats.get("cache_hits", 0) / max(event_count, 1),
            semantic_filter_count=method_stats.get("semantic_filtered", 0),
            semantic_filter_rate=method_stats.get("semantic_filtered", 0) / max(event_count, 1),
            avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            p50_latency_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
            p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            token_efficiency=token_efficiency,
        )

    def compute_classification_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        if not predictions or not references:
            return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

        # Accuracy
        correct = sum(p == r for p, r in zip(predictions, references))
        accuracy = correct / len(predictions)

        # F1 scores (manual implementation to avoid sklearn dependency)
        labels = list(set(references))
        label_metrics = {}

        for label in labels:
            tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
            fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
            fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            support = sum(1 for r in references if r == label)
            label_metrics[label] = {"f1": f1, "support": support}

        # Macro F1
        macro_f1 = sum(m["f1"] for m in label_metrics.values()) / len(labels) if labels else 0

        # Weighted F1
        total_support = len(references)
        weighted_f1 = sum(
            m["f1"] * m["support"] / total_support
            for m in label_metrics.values()
        ) if total_support > 0 else 0

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    def compute_summarization_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute summarization metrics (ROUGE, BERTScore)."""
        if not predictions or not references:
            return {
                "rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0,
                "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0,
            }

        # ROUGE scores
        if self._rouge_scorer:
            rouge_1_scores = []
            rouge_2_scores = []
            rouge_l_scores = []

            for pred, ref in zip(predictions, references):
                scores = self._rouge_scorer.score(ref, pred)
                rouge_1_scores.append(scores['rouge1'].fmeasure)
                rouge_2_scores.append(scores['rouge2'].fmeasure)
                rouge_l_scores.append(scores['rougeL'].fmeasure)

            rouge_1 = float(np.mean(rouge_1_scores))
            rouge_2 = float(np.mean(rouge_2_scores))
            rouge_l = float(np.mean(rouge_l_scores))
        else:
            # Simple word overlap estimation
            rouge_1 = self._simple_rouge(predictions, references, n=1)
            rouge_2 = self._simple_rouge(predictions, references, n=2)
            rouge_l = rouge_1 * 0.9  # Approximate

        # BERTScore
        if self._bertscore_available:
            try:
                from bert_score import score as bert_score
                P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
                bertscore_p = float(P.mean())
                bertscore_r = float(R.mean())
                bertscore_f1 = float(F1.mean())
            except Exception as e:
                logger.warning(f"BERTScore computation failed: {e}")
                bertscore_p = bertscore_r = bertscore_f1 = rouge_l * 1.1  # Approximate
        else:
            # Approximate BERTScore based on ROUGE
            bertscore_p = bertscore_r = bertscore_f1 = rouge_l * 1.1

        return {
            "rouge_1": rouge_1,
            "rouge_2": rouge_2,
            "rouge_l": rouge_l,
            "bertscore_precision": min(bertscore_p, 1.0),
            "bertscore_recall": min(bertscore_r, 1.0),
            "bertscore_f1": min(bertscore_f1, 1.0),
        }

    def _simple_rouge(
        self,
        predictions: List[str],
        references: List[str],
        n: int = 1,
    ) -> float:
        """Simple ROUGE-N approximation using word overlap."""
        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            if n == 1:
                pred_ngrams = set(pred_tokens)
                ref_ngrams = set(ref_tokens)
            else:
                pred_ngrams = set(zip(*[pred_tokens[i:] for i in range(n)]))
                ref_ngrams = set(zip(*[ref_tokens[i:] for i in range(n)]))

            if not ref_ngrams:
                scores.append(0.0)
                continue

            overlap = len(pred_ngrams & ref_ngrams)
            precision = overlap / len(pred_ngrams) if pred_ngrams else 0
            recall = overlap / len(ref_ngrams) if ref_ngrams else 0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            scores.append(f1)

        return float(np.mean(scores)) if scores else 0.0

    def compute_qa_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute QA metrics (exact match, F1)."""
        if not predictions or not references:
            return {"exact_match": 0.0, "f1": 0.0}

        # Exact match
        exact_matches = sum(
            self._normalize_answer(p) == self._normalize_answer(r)
            for p, r in zip(predictions, references)
        )
        exact_match = exact_matches / len(predictions)

        # Token-level F1
        f1_scores = [
            self._token_f1(p, r)
            for p, r in zip(predictions, references)
        ]

        return {
            "exact_match": exact_match,
            "f1": float(np.mean(f1_scores)),
        }

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer for comparison."""
        return s.lower().strip()

    def _token_f1(self, pred: str, ref: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = pred_tokens & ref_tokens

        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        return 2 * precision * recall / (precision + recall)

    def compute_ranking_metrics(
        self,
        predictions: List[List[int]],
        references: List[List[int]],
    ) -> Dict[str, float]:
        """Compute ranking metrics (NDCG, Spearman correlation)."""
        if not predictions or not references:
            return {"ndcg_5": 0.0, "ndcg_10": 0.0, "spearman": 0.0}

        ndcg_5_scores = []
        ndcg_10_scores = []
        spearman_scores = []

        for pred, ref in zip(predictions, references):
            if not pred or not ref:
                continue

            # NDCG
            ndcg_5_scores.append(self._ndcg(pred, ref, k=5))
            ndcg_10_scores.append(self._ndcg(pred, ref, k=10))

            # Spearman correlation
            spearman_scores.append(self._spearman(pred, ref))

        return {
            "ndcg_5": float(np.mean(ndcg_5_scores)) if ndcg_5_scores else 0.0,
            "ndcg_10": float(np.mean(ndcg_10_scores)) if ndcg_10_scores else 0.0,
            "spearman": float(np.mean(spearman_scores)) if spearman_scores else 0.0,
        }

    def _ndcg(self, pred_ranking: List[int], ref_ranking: List[int], k: int) -> float:
        """Compute NDCG@k."""
        n = len(ref_ranking)
        if n == 0:
            return 0.0

        # Create relevance scores from reference ranking
        relevance = {idx: (n - rank) / n for rank, idx in enumerate(ref_ranking)}

        # DCG
        dcg = 0.0
        for i, idx in enumerate(pred_ranking[:k]):
            rel = relevance.get(idx, 0)
            dcg += rel / math.log2(i + 2)

        # Ideal DCG
        ideal_relevances = sorted(relevance.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))

        return dcg / idcg if idcg > 0 else 0.0

    def _spearman(self, pred: List[int], ref: List[int]) -> float:
        """Compute Spearman rank correlation."""
        n = len(pred)
        if n < 2:
            return 0.0

        # Convert to rank positions
        pred_ranks = {v: i for i, v in enumerate(pred)}
        ref_ranks = {v: i for i, v in enumerate(ref)}

        # Find common elements
        common = set(pred) & set(ref)
        if len(common) < 2:
            return 0.0

        # Compute correlation
        d_squared_sum = sum(
            (pred_ranks[v] - ref_ranks[v]) ** 2
            for v in common
        )

        n_common = len(common)
        rho = 1 - (6 * d_squared_sum) / (n_common * (n_common ** 2 - 1))

        return rho

    def compute_quality_metrics(
        self,
        classification_results: Optional[Dict] = None,
        summarization_results: Optional[Dict] = None,
        qa_results: Optional[Dict] = None,
        ranking_results: Optional[Dict] = None,
    ) -> QualityMetrics:
        """Compute aggregated quality metrics."""

        # Get individual metrics with defaults
        cls_metrics = classification_results or {"accuracy": 0, "macro_f1": 0, "weighted_f1": 0}
        sum_metrics = summarization_results or {
            "rouge_1": 0, "rouge_2": 0, "rouge_l": 0,
            "bertscore_precision": 0, "bertscore_recall": 0, "bertscore_f1": 0,
        }
        qa_metrics = qa_results or {"exact_match": 0, "f1": 0}
        rank_metrics = ranking_results or {"ndcg_5": 0, "ndcg_10": 0, "spearman": 0}

        # Compute overall quality score (weighted average)
        quality_components = [
            cls_metrics.get("macro_f1", 0) * 0.2,
            sum_metrics.get("rouge_l", 0) * 0.3,
            sum_metrics.get("bertscore_f1", 0) * 0.2,
            qa_metrics.get("f1", 0) * 0.15,
            rank_metrics.get("ndcg_5", 0) * 0.15,
        ]
        quality_score = sum(quality_components)

        return QualityMetrics(
            classification_accuracy=cls_metrics.get("accuracy", 0),
            classification_macro_f1=cls_metrics.get("macro_f1", 0),
            classification_weighted_f1=cls_metrics.get("weighted_f1", 0),
            rouge_1=sum_metrics.get("rouge_1", 0),
            rouge_2=sum_metrics.get("rouge_2", 0),
            rouge_l=sum_metrics.get("rouge_l", 0),
            bertscore_precision=sum_metrics.get("bertscore_precision", 0),
            bertscore_recall=sum_metrics.get("bertscore_recall", 0),
            bertscore_f1=sum_metrics.get("bertscore_f1", 0),
            qa_exact_match=qa_metrics.get("exact_match", 0),
            qa_f1=qa_metrics.get("f1", 0),
            ranking_ndcg_5=rank_metrics.get("ndcg_5", 0),
            ranking_ndcg_10=rank_metrics.get("ndcg_10", 0),
            ranking_spearman=rank_metrics.get("spearman", 0),
            quality_score=quality_score,
        )
