from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_command_log(path: str | Path, command: Iterable[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(shlex.quote(str(part)) for part in command)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"```bash\n{command_text}\n```\n\n")
