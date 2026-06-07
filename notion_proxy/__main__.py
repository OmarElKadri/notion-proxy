"""CLI entry point: ``python -m notion_proxy``."""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .config import CONFIG_PATH


def main() -> None:
    ap = argparse.ArgumentParser(prog="notion_proxy")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()
    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
