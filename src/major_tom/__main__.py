"""Entry point: python -m major_tom"""

import argparse
import logging
import signal
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description="Major Tom Journal - AI-powered activity journaling")
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Run recorder only, without the web dashboard",
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Run web dashboard only, without the recorder",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web dashboard port (default: 8000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.web_only:
        _run_web(args.port)
        return

    if args.no_web:
        from major_tom.recorder import Major_Tom_Recorder
        Major_Tom_Recorder().run()
        return

    # Default: start both recorder and web dashboard
    _run_combined(args.port)


def _run_web(port: int) -> None:
    """Start only the web dashboard."""
    import uvicorn
    from major_tom.web.app import app  # noqa: F401 - triggers Config.load_config()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


def _run_combined(port: int) -> None:
    """Start recorder in background thread + web dashboard in main thread."""
    import uvicorn
    from major_tom.recorder import Major_Tom_Recorder

    # Import app FIRST so EventBus subscriptions are registered
    # before the recorder publishes its initial "running: True" event.
    from major_tom.web.app import app  # noqa: F401
    from major_tom.web.routers import metrics as metrics_router

    recorder = Major_Tom_Recorder()

    # Wire recorder's MetricsCollector to the web metrics endpoint
    metrics_router.set_collector(recorder.metrics_collector)

    # Run recorder in a daemon thread so it shuts down with the main process
    recorder_thread = threading.Thread(target=recorder.run, daemon=True, name="recorder")
    recorder_thread.start()

    logger = logging.getLogger(__name__)
    logger.info("Recorder started in background. Launching web dashboard on port %d ...", port)

    # Handle Ctrl+C: stop uvicorn, recorder thread will exit as daemon
    def _signal_handler(sig, frame):
        logger.info("Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
