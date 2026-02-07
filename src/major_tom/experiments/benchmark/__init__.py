"""Benchmark evaluation framework for ACM MM experiments."""

from major_tom.experiments.benchmark.dataset import (
    ActivityEvent,
    ActivitySession,
    SyntheticActivityDataset,
    BenchmarkDataLoader,
)
from major_tom.experiments.benchmark.metrics import (
    EfficiencyMetrics,
    QualityMetrics,
    MetricsCalculator,
)
from major_tom.experiments.benchmark.baselines import (
    BaselineMethod,
    NaiveFullBaseline,
    RuleBasedBaseline,
    FixedSamplingBaseline,
    EmbeddingOnlyBaseline,
    LLMClassifierBaseline,
)
from major_tom.experiments.benchmark.evaluator import BenchmarkEvaluator

__all__ = [
    "ActivityEvent",
    "ActivitySession",
    "SyntheticActivityDataset",
    "BenchmarkDataLoader",
    "EfficiencyMetrics",
    "QualityMetrics",
    "MetricsCalculator",
    "BaselineMethod",
    "NaiveFullBaseline",
    "RuleBasedBaseline",
    "FixedSamplingBaseline",
    "EmbeddingOnlyBaseline",
    "LLMClassifierBaseline",
    "BenchmarkEvaluator",
]
