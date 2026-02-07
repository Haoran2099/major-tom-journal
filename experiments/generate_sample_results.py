#!/usr/bin/env python3
"""
Generate sample experiment results for dashboard visualization.
This creates realistic-looking results without requiring Ollama to be running.
"""

import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure experiments/results directory exists
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def generate_token_efficiency_results():
    """Generate results for Token Efficiency experiments (te_c0 - te_c4)."""

    conditions = {
        "te_c0_naive": {
            "description": "Naive - No optimization",
            "semantic_enabled": False,
            "cache_enabled": False,
            "adaptive_enabled": False,
        },
        "te_c1_cache_only": {
            "description": "Cache Only",
            "semantic_enabled": False,
            "cache_enabled": True,
            "adaptive_enabled": False,
        },
        "te_c2_semantic_only": {
            "description": "Semantic Gating Only",
            "semantic_enabled": True,
            "cache_enabled": False,
            "adaptive_enabled": False,
        },
        "te_c3_sem_cache": {
            "description": "Semantic + Cache",
            "semantic_enabled": True,
            "cache_enabled": True,
            "adaptive_enabled": False,
        },
        "te_c4_full_system": {
            "description": "Full System (HSR)",
            "semantic_enabled": True,
            "cache_enabled": True,
            "adaptive_enabled": True,
        },
    }

    # Baseline tokens for naive method
    baseline_tokens = 45000

    results = []

    for name, config in conditions.items():
        # Calculate realistic token savings based on enabled features
        semantic_savings = 0.33 if config["semantic_enabled"] else 0
        cache_savings = 0.16 if config["cache_enabled"] else 0
        adaptive_savings = 0.08 if config["adaptive_enabled"] else 0

        total_savings = semantic_savings + cache_savings + adaptive_savings
        # Add some noise
        total_savings += random.uniform(-0.02, 0.02)
        total_savings = max(0, min(total_savings, 0.85))

        total_tokens = int(baseline_tokens * (1 - total_savings))

        # Calculate other metrics
        total_decisions = 180
        semantic_hit_rate = 0.53 if config["semantic_enabled"] else 0
        cache_hit_rate = 0.21 if config["cache_enabled"] else 0
        llm_brain_rate = 1 - semantic_hit_rate - cache_hit_rate

        result = {
            "experiment": {
                "name": name,
                "dimension": "token_efficiency",
                "run_id": 0,
                "trace": "traces/mixed_60m",
                "started_at": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                "ended_at": datetime.now().isoformat(),
                "duration_seconds": 3600 + random.randint(-100, 100),
                "config_hash": f"sha256:{random.randbytes(8).hex()}",
            },
            "models": {
                "brain": "qwen3:8b",
                "eye": "qwen3-vl:8b",
                "embedding": "qwen3-embedding:8b",
            },
            "components": {
                "semantic_gating": config["semantic_enabled"],
                "decision_cache": config["cache_enabled"],
                "vlm": True,
                "adaptive_sampling": config["adaptive_enabled"],
            },
            "token_metrics": {
                "total_tokens": total_tokens,
                "brain_prompt_tokens": int(total_tokens * 0.62),
                "brain_completion_tokens": int(total_tokens * 0.12),
                "eye_prompt_tokens": int(total_tokens * 0.18),
                "eye_completion_tokens": int(total_tokens * 0.04),
                "embedding_tokens": int(total_tokens * 0.04),
                "tokens_per_decision_mean": total_tokens / total_decisions,
            },
            "routing_metrics": {
                "total_decisions": total_decisions,
                "snapshot_count": int(total_decisions * 0.36),
                "skip_count": int(total_decisions * 0.64),
                "semantic_hit_count": int(total_decisions * semantic_hit_rate),
                "semantic_hit_rate": semantic_hit_rate,
                "cache_hit_count": int(total_decisions * cache_hit_rate),
                "cache_hit_rate": cache_hit_rate,
                "llm_brain_call_count": int(total_decisions * llm_brain_rate),
                "llm_brain_call_rate": llm_brain_rate,
                "vlm_call_count": 52 - int(20 * total_savings),
                "vlm_effective_rate": 0.75 + total_savings * 0.1,
            },
            "latency_metrics": {
                "decision_avg_ms": 85 + random.uniform(-10, 10),
                "decision_p50_ms": 12 + random.uniform(-2, 2),
                "decision_p95_ms": 1850 - total_savings * 500,
                "semantic_avg_ms": 8.5 if config["semantic_enabled"] else 0,
                "brain_avg_ms": 1200 + random.uniform(-100, 100),
                "vlm_avg_ms": 3500 + random.uniform(-200, 200),
            },
            "efficiency": {
                "token_savings_vs_naive": total_savings,
                "semantic_layer_contribution": semantic_savings,
                "cache_layer_contribution": cache_savings,
                "adaptive_contribution": adaptive_savings,
            },
        }

        results.append(result)

    return results


def generate_memory_mechanism_results():
    """Generate results for Memory Mechanism experiments (mm_m0 - mm_m3)."""

    conditions = {
        "mm_m0_global": {
            "description": "Global Memory",
            "context_pollution_rate": 0.35,
            "context_recall_accuracy": 0.62,
            "memory_isolation_score": 0.45,
        },
        "mm_m1_app_only": {
            "description": "App-Only Routing",
            "context_pollution_rate": 0.22,
            "context_recall_accuracy": 0.71,
            "memory_isolation_score": 0.68,
        },
        "mm_m2_keyword": {
            "description": "Keyword Routing",
            "context_pollution_rate": 0.08,
            "context_recall_accuracy": 0.85,
            "memory_isolation_score": 0.88,
        },
        "mm_m3_semantic": {
            "description": "Semantic Routing",
            "context_pollution_rate": 0.05,
            "context_recall_accuracy": 0.91,
            "memory_isolation_score": 0.93,
        },
    }

    results = []

    for name, config in conditions.items():
        # Add some noise
        cpr = config["context_pollution_rate"] + random.uniform(-0.02, 0.02)
        cra = config["context_recall_accuracy"] + random.uniform(-0.02, 0.02)
        mis = config["memory_isolation_score"] + random.uniform(-0.02, 0.02)

        result = {
            "experiment": {
                "name": name,
                "dimension": "memory_mechanism",
                "run_id": 0,
                "trace": "traces/mixed_60m",
                "started_at": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                "ended_at": datetime.now().isoformat(),
                "duration_seconds": 3600 + random.randint(-100, 100),
            },
            "models": {
                "brain": "qwen3:8b",
                "embedding": "qwen3-embedding:8b",
            },
            "components": {
                "memory_type": name.split("_")[-1],
            },
            "memory_metrics": {
                "task_switch_count": 15 + random.randint(-3, 3),
                "unique_task_ids": 6,
                "memory_files_created": 6 if "m0" not in name else 1,
                "context_pollution_rate": max(0, min(1, cpr)),
                "context_recall_accuracy": max(0, min(1, cra)),
                "memory_isolation_score": max(0, min(1, mis)),
                "decision_consistency": 0.75 + (1 - cpr) * 0.2,
            },
            "switch_patterns": {
                "simple_return": {"count": 5, "cpr": cpr * 0.8},
                "multi_hop_return": {"count": 3, "cpr": cpr * 1.1},
                "frequent_alternation": {"count": 4, "cpr": cpr * 1.2},
                "deep_stack": {"count": 2, "cpr": cpr * 1.3},
            },
        }

        results.append(result)

    return results


def generate_journal_quality_results():
    """Generate results for Journal Quality experiments (jq_q0 - jq_q4)."""

    conditions = {
        "jq_q0_small": {
            "description": "Small Models (4b)",
            "brain": "qwen3:4b",
            "eye": "qwen3-vl:4b",
            "quality_base": 0.65,
        },
        "jq_q1_baseline": {
            "description": "Baseline (8b)",
            "brain": "qwen3:8b",
            "eye": "qwen3-vl:8b",
            "quality_base": 0.78,
        },
        "jq_q2_large": {
            "description": "Large Models (14b)",
            "brain": "qwen3:14b",
            "eye": "qwen3-vl:14b",
            "quality_base": 0.85,
        },
        "jq_q3_gemma": {
            "description": "Gemma Models (12b)",
            "brain": "gemma3:12b",
            "eye": "gemma3-vl:12b",
            "quality_base": 0.82,
        },
        "jq_q4_no_vlm": {
            "description": "Text Only (No VLM)",
            "brain": "qwen3:8b",
            "eye": None,
            "quality_base": 0.68,
        },
    }

    results = []

    for name, config in conditions.items():
        q = config["quality_base"]

        result = {
            "experiment": {
                "name": name,
                "dimension": "journal_quality",
                "run_id": 0,
                "trace": "traces/mixed_60m",
                "started_at": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                "ended_at": datetime.now().isoformat(),
                "duration_seconds": 3600 + random.randint(-100, 100),
            },
            "models": {
                "brain": config["brain"],
                "eye": config["eye"],
                "embedding": "qwen3-embedding:8b",
            },
            "components": {
                "vlm_enabled": config["eye"] is not None,
            },
            "journal_quality": {
                "relevance_score_mean": q + random.uniform(-0.03, 0.03),
                "redundancy_rate": 0.35 - q * 0.2 + random.uniform(-0.02, 0.02),
                "activity_coverage_rate": q * 1.05 + random.uniform(-0.03, 0.03),
                "information_density_mean": 2.5 + q * 1.5 + random.uniform(-0.2, 0.2),
                "entry_count": 52 + random.randint(-5, 5),
                "entry_length_mean_chars": 120 + q * 50 + random.uniform(-10, 10),
            },
            "token_metrics": {
                "total_tokens": int(30000 + (q - 0.65) * 50000),
                "tokens_per_entry": int(600 + (q - 0.65) * 400),
            },
            "human_evaluation": {
                "accuracy": q + random.uniform(-0.05, 0.05),
                "informativeness": q * 0.95 + random.uniform(-0.05, 0.05),
                "conciseness": 0.9 - (q - 0.65) * 0.3 + random.uniform(-0.05, 0.05),
                "actionability": q * 0.9 + random.uniform(-0.05, 0.05),
            },
        }

        results.append(result)

    return results


def main():
    """Generate all sample results."""
    random.seed(42)

    print("Generating sample experiment results...")

    # Token Efficiency
    te_results = generate_token_efficiency_results()
    for r in te_results:
        name = r["experiment"]["name"]
        output_dir = RESULTS_DIR / name
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"  ✓ {name}")

    # Memory Mechanism
    mm_results = generate_memory_mechanism_results()
    for r in mm_results:
        name = r["experiment"]["name"]
        output_dir = RESULTS_DIR / name
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"  ✓ {name}")

    # Journal Quality
    jq_results = generate_journal_quality_results()
    for r in jq_results:
        name = r["experiment"]["name"]
        output_dir = RESULTS_DIR / name
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"  ✓ {name}")

    # Summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "experiments": {
            "token_efficiency": [r["experiment"]["name"] for r in te_results],
            "memory_mechanism": [r["experiment"]["name"] for r in mm_results],
            "journal_quality": [r["experiment"]["name"] for r in jq_results],
        },
        "total_experiments": len(te_results) + len(mm_results) + len(jq_results),
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Generated {summary['total_experiments']} experiment results")
    print(f"  Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
