#!/usr/bin/env python3
"""
Run takemeter.ipynb sections 6–7 (three seeds, confusion matrix) headless. ← UNIT 6

    python run_three_seeds.py

Executes the notebook's own cells, in order, in one namespace, from this folder:
sections 1–3 as written, then only the imports and function definitions from
the section 4 and 5 cells (so the single-seed Unit 5 run does not happen again
and results.json / test_split.csv are not rewritten), then sections 6 and 7 as
written. Nothing is reimplemented here: train_model, split_dataset, evaluate
and the section 7 helpers are the notebook's.

Section 6 writes results_three_seeds_<RUN_LABEL>.json. The script refuses to
run if that file already exists, and checks results.json, test_split.csv and
(on the "after" run) results_three_seeds_before.json are byte-identical
afterwards.
"""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass

NOTEBOOK = Path("takemeter.ipynb")
STOP_MARKER = "Unit 6 starts here"
PROTECTED = [Path("results.json"), Path("test_split.csv"), Path("results_three_seeds_before.json")]


def definitions_only(source):
    """Keep a cell's imports and function definitions; drop everything it runs."""
    tree = ast.parse(source)
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef)
    tree.body = [node for node in tree.body if isinstance(node, keep)]
    return ast.unparse(tree)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cells = [c for c in json.loads(NOTEBOOK.read_text())["cells"]]
    stop = next(i for i, c in enumerate(cells) if STOP_MARKER in "".join(c["source"]))
    code = [(i, "".join(c["source"])) for i, c in enumerate(cells) if c["cell_type"] == "code"]

    run_label = re.search(r'^RUN_LABEL = "(\w+)"', "".join(src for _, src in code), re.M).group(1)
    out = Path(f"results_three_seeds_{run_label}.json")
    if out.exists():
        print(f"{out} already exists. Move it aside first; not overwriting.", file=sys.stderr)
        return 1

    protected = [p for p in PROTECTED if p.exists() and p != out]
    before = {p: digest(p) for p in protected}
    namespace = {"__name__": "__main__"}

    for i, source in code:
        trains = "train_model(" in source and "def train_model" in source
        writes = "results.json" in source and "def evaluate" in source
        if i < stop and (trains or writes):
            source = definitions_only(source)
            print(f"\n── cell {i}: definitions only ──", flush=True)
        else:
            print(f"\n── cell {i} ──", flush=True)
        exec(compile(source, f"<takemeter.ipynb cell {i}>", "exec"), namespace)

    changed = [str(p) for p in protected if digest(p) != before[p]]
    if changed:
        print(f"\n[ERROR] changed during the run: {', '.join(changed)}", file=sys.stderr)
        return 1
    print(f"\n{', '.join(map(str, protected))} unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
