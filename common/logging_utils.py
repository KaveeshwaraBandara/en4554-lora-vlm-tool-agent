"""Tiny logging helpers: one JSON object per line, flushed immediately.

Flushing on every write means a Colab disconnect loses at most the current step.
"""

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path, append: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a" if append else "w", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        record = {"time": time.time(), **record}
        self._f.write(json.dumps(record, default=_to_jsonable) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_jsonable(x: Any) -> Any:
    # torch / numpy scalars and Paths
    if hasattr(x, "item"):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return str(x)
