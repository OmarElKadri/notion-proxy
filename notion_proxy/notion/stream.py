"""Parse Notion runInferenceTranscript NDJSON patch streams into text deltas."""

from __future__ import annotations

from typing import Any, Iterable


class NotionPatchStream:
    """Parses runInferenceTranscript NDJSON and yields text deltas."""

    def __init__(self, include_thinking: bool = False) -> None:
        self.steps: list[Any] = []
        self.include_thinking = include_thinking

    def feed(self, obj: dict) -> Iterable[str]:
        t = obj.get("type")
        if t == "patch-start":
            self.steps = list(obj.get("data", {}).get("s", []))
            for step in self.steps:
                yield from self._initial_text(step)
            return
        if t == "patch":
            for op in obj.get("v", []) or []:
                yield from self._apply_op(op)
            return
        if t == "patch-sync":
            self.steps = list(obj.get("data", {}).get("s", []))
            return

    def _initial_text(self, step: dict) -> Iterable[str]:
        if not isinstance(step, dict):
            return
        if step.get("type") != "agent-inference":
            return
        for part in step.get("value", []) or []:
            kind = part.get("type")
            content = part.get("content") or ""
            if kind == "text" and content:
                yield content
            elif kind == "thinking" and content and self.include_thinking:
                yield content

    def _apply_op(self, op: dict) -> Iterable[str]:
        o = op.get("o")
        p = op.get("p") or ""
        v = op.get("v")
        segs = [s for s in p.split("/") if s]
        if not segs or segs[0] != "s":
            return

        if o == "a":
            if len(segs) == 2 and segs[1] == "-":
                self.steps.append(v)
                yield from self._initial_text(v if isinstance(v, dict) else {})
                return
            if len(segs) >= 4 and segs[2] == "value" and segs[3] == "-":
                try:
                    n = int(segs[1])
                    step = self.steps[n]
                    step.setdefault("value", []).append(v)
                except (IndexError, ValueError, AttributeError):
                    return
                if isinstance(v, dict):
                    kind = v.get("type")
                    content = v.get("content") or ""
                    if kind == "text" and content:
                        yield content
                    elif kind == "thinking" and content and self.include_thinking:
                        yield content
                return
            self._set_path(segs[1:], v)
            return

        if o == "x":
            if len(segs) == 5 and segs[2] == "value" and segs[4] == "content":
                try:
                    n = int(segs[1])
                    i = int(segs[3])
                    part = self.steps[n]["value"][i]
                except (IndexError, KeyError, ValueError, TypeError):
                    return
                part["content"] = (part.get("content") or "") + (v or "")
                kind = part.get("type")
                if v and kind == "text":
                    yield v
                elif v and kind == "thinking" and self.include_thinking:
                    yield v
                return
            self._append_path(segs[1:], v if isinstance(v, str) else "")
            return

    def _set_path(self, rel: list[str], value: Any) -> None:
        if not rel:
            return
        try:
            node: Any = self.steps[int(rel[0])]
        except (IndexError, ValueError):
            return
        for s in rel[1:-1]:
            node = node[int(s)] if isinstance(node, list) else node.setdefault(s, {})
        last = rel[-1]
        if isinstance(node, list):
            try:
                node[int(last)] = value
            except (IndexError, ValueError):
                pass
        elif isinstance(node, dict):
            node[last] = value

    def _append_path(self, rel: list[str], value: str) -> None:
        if not rel:
            return
        try:
            node: Any = self.steps[int(rel[0])]
        except (IndexError, ValueError):
            return
        for s in rel[1:-1]:
            node = node[int(s)] if isinstance(node, list) else node.get(s, {})
        last = rel[-1]
        if isinstance(node, dict) and isinstance(node.get(last), str):
            node[last] = node[last] + value
