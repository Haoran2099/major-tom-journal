#!/usr/bin/env python3
"""
ACM MM Benchmark: One-click experiment runner.

Usage:
    # Generate dataset and run all experiments
    python -m major_tom.experiments.acmmm.run_all_experiments

    # Run with existing dataset
    python -m major_tom.experiments.acmmm.run_all_experiments --dataset experiments/datasets/synthetic

    # Run specific methods only
    python -m major_tom.experiments.acmmm.run_all_experiments --methods naive_full,rule_based,hsr

    # Quick test with small dataset
    python -m major_tom.experiments.acmmm.run_all_experiments --quick
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from major_tom.experiments.benchmark.dataset import SyntheticActivityDataset
from major_tom.experiments.benchmark.evaluator import BenchmarkEvaluator
from major_tom.experiments.benchmark.baselines import BASELINE_METHODS

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def generate_dataset(output_dir: Path, num_sessions: int, events_per_session: int, seed: int):
    """Generate synthetic dataset."""
    logger.info(f"Generating synthetic dataset: {num_sessions} sessions, {events_per_session} events each")

    dataset = SyntheticActivityDataset(output_dir, seed=seed)
    dataset.generate(
        num_sessions=num_sessions,
        events_per_session=events_per_session,
        generate_summaries=True,
        generate_qa=True,
    )

    logger.info(f"Dataset saved to {output_dir}")
    return output_dir


def run_benchmark(
    dataset_path: Path,
    output_dir: Path,
    methods: list,
    use_llm: bool = False,
):
    """Run benchmark evaluation."""
    logger.info(f"Running benchmark with methods: {methods}")

    # Initialize LLM backend if requested
    llm_backend = None
    if use_llm:
        try:
            from major_tom.llm.ollama_backend import OllamaBackend
            llm_backend = OllamaBackend()
            logger.info("Using Ollama LLM backend")
        except Exception as e:
            logger.warning(f"Could not initialize LLM backend: {e}")
            logger.warning("Running without LLM support (using heuristics)")

    # Run evaluation
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        output_dir=output_dir,
        llm_backend=llm_backend,
    )

    results = evaluator.run_all_methods(methods=methods)

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(evaluator.generate_comparison_table())

    print("\n" + "-" * 70)
    print("LaTeX Table for Paper:")
    print("-" * 70)
    print(evaluator.generate_latex_table())

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ACM MM Benchmark: Run all experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full benchmark (generates dataset + runs all methods)
    python -m major_tom.experiments.acmmm.run_all_experiments

    # Quick test (10 sessions, 20 events each)
    python -m major_tom.experiments.acmmm.run_all_experiments --quick

    # Use existing dataset
    python -m major_tom.experiments.acmmm.run_all_experiments \\
        --dataset experiments/datasets/synthetic

    # Run with LLM backend (requires Ollama)
    python -m major_tom.experiments.acmmm.run_all_experiments --use-llm

    # Run specific methods only
    python -m major_tom.experiments.acmmm.run_all_experiments \\
        --methods naive_full,rule_based,hsr
        """,
    )

    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        help="Path to existing dataset. If not provided, generates new dataset.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("experiments/results/acmmm"),
        help="Output directory for results (default: experiments/results/acmmm)",
    )
    parser.add_argument(
        "--methods", "-m",
        type=str,
        default="all",
        help=f"Comma-separated list of methods, or 'all'. Available: {', '.join(BASELINE_METHODS.keys())}",
    )
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=100,
        help="Number of sessions to generate (default: 100)",
    )
    parser.add_argument(
        "--events-per-session",
        type=int,
        default=100,
        help="Events per session (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode: 10 sessions, 20 events each",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM backend (requires Ollama running)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Quick mode overrides
    if args.quick:
        args.num_sessions = 10
        args.events_per_session = 20
        logger.info("Quick mode: 10 sessions, 20 events each")

    # Parse methods
    if args.methods == "all":
        methods = list(BASELINE_METHODS.keys())
    else:
        methods = [m.strip() for m in args.methods.split(",")]
        # Validate methods
        for m in methods:
            if m not in BASELINE_METHODS:
                logger.error(f"Unknown method: {m}. Available: {list(BASELINE_METHODS.keys())}")
                sys.exit(1)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate or use existing dataset
    if args.dataset:
        dataset_path = args.dataset
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            sys.exit(1)
    else:
        dataset_path = output_dir / "dataset"
        generate_dataset(
            dataset_path,
            num_sessions=args.num_sessions,
            events_per_session=args.events_per_session,
            seed=args.seed,
        )

    # Run benchmark
    results = run_benchmark(
        dataset_path=dataset_path,
        output_dir=output_dir,
        methods=methods,
        use_llm=args.use_llm,
    )

    # Save run configuration
    config = {
        "timestamp": timestamp,
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "methods": methods,
        "num_sessions": args.num_sessions,
        "events_per_session": args.events_per_session,
        "seed": args.seed,
        "use_llm": args.use_llm,
    }

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
