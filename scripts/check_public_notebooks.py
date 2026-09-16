"""Fast, data-free checks for notebooks intended for the public repository."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
PRIVATE_PATTERNS = {
    "Linux home path": re.compile(r"/home/[^/\s]+/"),
    "Windows absolute path": re.compile(r"\b[A-Za-z]:[\\/]"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "credential assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|password|secret|access[_-]?token)\s*=\s*['\"][^'\"]+['\"]"
    ),
}


def main() -> None:
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not paths:
        raise SystemExit("No public notebooks found.")

    failures: list[str] = []
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", []), start=1):
            source = cell.get("source", "")
            source = "".join(source) if isinstance(source, list) else str(source)
            for label, pattern in PRIVATE_PATTERNS.items():
                if pattern.search(source):
                    failures.append(f"{path}: cell {index} contains a {label}")
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs"):
                failures.append(f"{path}: cell {index} contains saved output")
            if cell.get("execution_count") is not None:
                failures.append(f"{path}: cell {index} contains an execution count")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Checked {len(paths)} output-free public notebooks.")


if __name__ == "__main__":
    main()
