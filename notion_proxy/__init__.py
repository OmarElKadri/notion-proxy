"""Notion AI -> Anthropic Messages API proxy.

``create_app`` is imported lazily so that importing pure-logic submodules
(e.g. ``notion_proxy.tools.parse``) does not pull in the HTTP stack
(``curl_cffi``) or FastAPI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["create_app"]

if TYPE_CHECKING:
    from .app import create_app


def __getattr__(name: str):
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
