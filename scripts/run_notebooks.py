"""Execute every notebook in order, in place, so figures/ and results/ are regenerated.

Usage (from the repository root):

    python scripts/run_notebooks.py            # all notebooks
    python scripts/run_notebooks.py 03 06      # only notebooks whose name starts with 03 / 06

Notebook 06 reads the CSV files written by 01-05, so run those first.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def main(prefixes: list[str]) -> None:
    notebooks = sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb"))
    if prefixes:
        notebooks = [nb for nb in notebooks if any(nb.name.startswith(p) for p in prefixes)]
    for path in notebooks:
        t0 = time.perf_counter()
        print(f"▶ {path.name} ...", flush=True)
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(NB_DIR)}}).execute()
        nbformat.write(nb, path)
        print(f"  done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
