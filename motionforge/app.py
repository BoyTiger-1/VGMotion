"""Application entry point: `python -m motionforge [--selftest] [--dry-run]`."""
from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    parser = argparse.ArgumentParser(prog="motionforge",
                                     description="Universal AI motion controls for PC games")
    parser.add_argument("--selftest", action="store_true",
                        help="run the end-to-end pipeline self-test and exit")
    parser.add_argument("--no-camera-test", action="store_true",
                        help="selftest: skip the live camera stage")
    parser.add_argument("--dry-run", action="store_true",
                        help="start with input injection disabled (log only)")
    parser.add_argument("--camera", type=int, default=None, help="camera index override")
    args = parser.parse_args(argv)

    if args.selftest:
        from motionforge.core.selftest import run_selftest
        return run_selftest(camera=not args.no_camera_test)

    from PySide6.QtWidgets import QApplication
    from motionforge import config
    from motionforge.core.engine import MotionEngine
    from motionforge.ui.mainwindow import MainWindow

    config.ensure_app_dirs()
    settings = config.Settings()
    if args.dry_run:
        settings.set("dry_run", True)
    if args.camera is not None:
        settings.set("camera_indices", [args.camera])

    app = QApplication(sys.argv[:1])
    app.setApplicationName("MotionForge")
    engine = MotionEngine(settings)
    window = MainWindow(engine)
    window.show()
    engine.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
