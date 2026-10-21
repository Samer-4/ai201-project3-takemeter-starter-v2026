#!/usr/bin/env python3
"""
Run takemeter.ipynb sections 1–5 headless, in this project's .venv. ← UNIT 5

    python run_notebook.py

The same cells you would run by hand in VS Code, in order, in one kernel, from
this folder — so `results.json` and `test_split.csv` land here. Execution stops
at the "Unit 6 starts here" marker; nothing below it runs. Outputs are written
back into the notebook so the committed file shows what the run printed.

Needs `nbclient` and `ipykernel` (`pip install nbclient ipykernel`); they are
tooling for running the notebook, not dependencies of it, so requirements.txt
doesn't pin them.
"""

import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass

NOTEBOOK = Path("takemeter.ipynb")
STOP_MARKER = "Unit 6 starts here"


def main():
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        print("Needs nbclient: pip install nbclient ipykernel", file=sys.stderr)
        return 1

    nb = nbformat.read(NOTEBOOK, as_version=4)
    stop = next((i for i, c in enumerate(nb.cells) if STOP_MARKER in c.source), len(nb.cells))
    unit5 = nbformat.v4.new_notebook(metadata=nb.metadata, cells=nb.cells[:stop])

    print(f"Running cells 0–{stop - 1} of {NOTEBOOK} (sections 1–5)…", flush=True)
    started = time.time()
    client = NotebookClient(unit5, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(Path.cwd())}})
    # Use this interpreter — the .venv — rather than whatever "python3" resolves to.
    client.kernel_manager_class = "jupyter_client.manager.KernelManager"
    client.extra_arguments = []
    km = client.create_kernel_manager()
    km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        client.execute(km=km)
    finally:
        # Keep whatever ran, even on failure, so the error is visible in the file.
        nb.cells[:stop] = unit5.cells
        nbformat.write(nb, NOTEBOOK)

    print(f"Done in {time.time() - started:.0f}s. Outputs saved into {NOTEBOOK}.")
    for cell in unit5.cells:
        for out in cell.get("outputs", []):
            if out.get("name") == "stdout":
                print(out["text"], end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
