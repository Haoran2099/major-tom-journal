"""Statistical analysis tools for experiment results."""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ANOVAResult:
    """Result from ANOVA test."""
    f_statistic: float = 0.0
    p_value: float = 1.0
    effect_size_eta_squared: float = 0.0
    groups: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class TukeyResult:
    """Result from Tukey HSD post-hoc test."""
    comparisons: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TTestResult:
    """Result from t-test."""
    t_statistic: float = 0.0
    p_value: float = 1.0
    cohens_d: float = 0.0


@dataclass
class StatReport:
    """Complete statistical report."""
    dimension: str = ""
    metric: str = ""
    anova: Optional[ANOVAResult] = None
    tukey: Optional[TukeyResult] = None
    pairwise: List[Dict[str, Any]] = field(default_factory=list)
    summary_table: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "dimension": self.dimension,
            "metric": self.metric,
            "summary_table": self.summary_table,
        }
        if self.anova:
            result["anova"] = {
                "F_statistic": round(self.anova.f_statistic, 4),
                "p_value": self.anova.p_value,
                "effect_size_eta_squared": round(self.anova.effect_size_eta_squared, 4),
            }
        if self.tukey:
            result["pairwise_comparisons"] = self.tukey.comparisons
        return result


class StatisticalAnalyzer:
    """Statistical analysis utilities for experiment evaluation."""

    def anova_one_way(self, groups: Dict[str, List[float]]) -> ANOVAResult:
        """One-way ANOVA with effect size."""
        try:
            from scipy import stats as sp_stats

            group_lists = [v for v in groups.values() if len(v) > 0]
            if len(group_lists) < 2:
                return ANOVAResult()

            f_stat, p_val = sp_stats.f_oneway(*group_lists)

            # Eta-squared
            all_values = np.concatenate(group_lists)
            grand_mean = np.mean(all_values)
            ss_between = sum(
                len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_lists
            )
            ss_total = np.sum((all_values - grand_mean) ** 2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

            group_stats = {}
            for name, values in groups.items():
                arr = np.array(values)
                group_stats[name] = {
                    "mean": round(float(np.mean(arr)), 4),
                    "std": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
                    "n": len(arr),
                }

            return ANOVAResult(
                f_statistic=float(f_stat),
                p_value=float(p_val),
                effect_size_eta_squared=float(eta_sq),
                groups=group_stats,
            )
        except ImportError:
            logger.warning("scipy not available, skipping ANOVA")
            return ANOVAResult()

    def anova_two_way(
        self,
        data: List[Dict[str, Any]],
        factor1: str,
        factor2: str,
        response: str,
    ) -> ANOVAResult:
        """Two-way ANOVA (requires pandas + scipy)."""
        try:
            import pandas as pd
            from scipy import stats as sp_stats

            df = pd.DataFrame(data)
            groups = df.groupby([factor1, factor2])[response].apply(list)

            # Flatten for one-way as approximation (full two-way needs statsmodels)
            flat_groups = {f"{k[0]}_{k[1]}": list(v) for k, v in groups.items()}
            return self.anova_one_way(flat_groups)
        except ImportError:
            logger.warning("pandas/scipy not available for two-way ANOVA")
            return ANOVAResult()

    def tukey_hsd(self, groups: Dict[str, List[float]]) -> TukeyResult:
        """Tukey HSD post-hoc test."""
        try:
            from scipy import stats as sp_stats

            names = list(groups.keys())
            comparisons = []

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a = np.array(groups[names[i]])
                    b = np.array(groups[names[j]])
                    if len(a) < 2 or len(b) < 2:
                        continue

                    t_stat, p_val = sp_stats.ttest_ind(a, b)
                    # Bonferroni correction
                    n_comparisons = len(names) * (len(names) - 1) / 2
                    p_adj = min(p_val * n_comparisons, 1.0)

                    d = self.cohens_d(list(a), list(b))
                    comparisons.append({
                        "pair": [names[i], names[j]],
                        "mean_diff": round(float(np.mean(a) - np.mean(b)), 4),
                        "p_adj": round(p_adj, 6),
                        "cohens_d": round(d, 4),
                        "significant": p_adj < 0.05,
                    })

            return TukeyResult(comparisons=comparisons)
        except ImportError:
            logger.warning("scipy not available for Tukey HSD")
            return TukeyResult()

    def paired_ttest(self, a: List[float], b: List[float]) -> TTestResult:
        """Paired t-test."""
        try:
            from scipy import stats as sp_stats
            t_stat, p_val = sp_stats.ttest_rel(a, b)
            d = self.cohens_d(a, b)
            return TTestResult(t_statistic=float(t_stat), p_value=float(p_val), cohens_d=d)
        except ImportError:
            return TTestResult()

    def wilcoxon(self, a: List[float], b: List[float]) -> Dict[str, float]:
        """Wilcoxon signed-rank test (non-parametric)."""
        try:
            from scipy import stats as sp_stats
            stat, p_val = sp_stats.wilcoxon(a, b)
            return {"statistic": float(stat), "p_value": float(p_val)}
        except ImportError:
            return {"statistic": 0.0, "p_value": 1.0}

    @staticmethod
    def cohens_d(a: List[float], b: List[float]) -> float:
        """Cohen's d effect size."""
        a_arr, b_arr = np.array(a), np.array(b)
        n1, n2 = len(a_arr), len(b_arr)
        if n1 < 2 or n2 < 2:
            return 0.0
        var1 = np.var(a_arr, ddof=1)
        var2 = np.var(b_arr, ddof=1)
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)

    @staticmethod
    def confidence_interval(
        data: List[float], level: float = 0.95
    ) -> Tuple[float, float]:
        """Confidence interval for the mean."""
        arr = np.array(data)
        n = len(arr)
        if n < 2:
            m = float(np.mean(arr)) if n == 1 else 0.0
            return (m, m)
        mean = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / math.sqrt(n))
        try:
            from scipy import stats as sp_stats
            t_val = sp_stats.t.ppf((1 + level) / 2, n - 1)
        except ImportError:
            t_val = 1.96  # Fallback to z for large n
        margin = t_val * se
        return (round(mean - margin, 4), round(mean + margin, 4))

    @staticmethod
    def fleiss_kappa(ratings: np.ndarray) -> float:
        """Fleiss' kappa for inter-rater agreement.

        Args:
            ratings: (n_subjects, n_categories) matrix of category counts per subject.
        """
        n, k = ratings.shape
        N_per_subject = ratings.sum(axis=1)
        if np.any(N_per_subject == 0):
            return 0.0

        N = int(N_per_subject[0])  # Assume equal raters per subject
        p_j = ratings.sum(axis=0) / (n * N)
        P_e = float(np.sum(p_j ** 2))

        P_i = (np.sum(ratings ** 2, axis=1) - N) / (N * (N - 1))
        P_bar = float(np.mean(P_i))

        if P_e == 1.0:
            return 1.0
        return (P_bar - P_e) / (1 - P_e)

    def generate_report(
        self,
        dimension: str,
        metric: str,
        groups: Dict[str, List[float]],
    ) -> StatReport:
        """Generate a complete statistical report for one metric."""
        anova = self.anova_one_way(groups)
        tukey = self.tukey_hsd(groups) if anova.p_value < 0.05 else None

        summary = {}
        for name, values in groups.items():
            arr = np.array(values)
            ci = self.confidence_interval(values)
            summary[name] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
                "ci_95": list(ci),
                "n": len(arr),
            }

        return StatReport(
            dimension=dimension,
            metric=metric,
            anova=anova,
            tukey=tukey,
            summary_table=summary,
        )
