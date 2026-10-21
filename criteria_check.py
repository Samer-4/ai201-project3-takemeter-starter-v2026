#!/usr/bin/env python3
"""
Score the five criteria in criteria.md against the files the run produced.

    python criteria_check.py

Reads results.json (the notebook, section 5), labels.csv, and
data/cold20_agreement.json (labelling.py agree). Prints a MET / MISSED table
ready to paste and writes criteria_results.json. The targets are copied from
criteria.md as filed; this script only does the arithmetic.

Unit 5 runs one seed (42). Unit 6 asks for each target to hold across three
seeds, which this doesn't test.
"""

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass

TARGETS = {
    "accuracy": 0.75,
    "f1": {"question": 0.80, "shared_info": 0.60, "personal_update": 0.50},
    "largest_share_max": 0.65,
    "smallest_share_min": 0.15,
    "cold_agreement_min": 15,
    "confidence_gap_min": 0.15,
}


def evaluate(results, label_counts, agreement):
    """Returns a list of (number, criterion, target, actual, met)."""
    rows = []
    rows.append((1, "Test accuracy", f">= {TARGETS['accuracy']:.2f}",
                 f"{results['accuracy']:.3f}", results["accuracy"] >= TARGETS["accuracy"]))

    f1 = {label: results["per_label"][label]["f1"] for label in TARGETS["f1"]}
    rows.append((2, "Per-label F1",
                 ", ".join(f"{l} >= {t:.2f}" for l, t in TARGETS["f1"].items()),
                 ", ".join(f"{l} {f1[l]:.3f}" for l in TARGETS["f1"]),
                 all(f1[l] >= t for l, t in TARGETS["f1"].items())))

    total = sum(label_counts.values())
    largest = max(label_counts.values()) / total
    smallest = min(label_counts.values()) / total
    rows.append((3, "Balance in labels.csv",
                 f"largest <= {TARGETS['largest_share_max']:.0%}, smallest >= {TARGETS['smallest_share_min']:.0%}",
                 f"largest {largest:.1%}, smallest {smallest:.1%}",
                 largest <= TARGETS["largest_share_max"] and smallest >= TARGETS["smallest_share_min"]))

    rows.append((4, "Cold-20 agreement (human vs blind AI)",
                 f">= {TARGETS['cold_agreement_min']}/20",
                 f"{agreement['agreed']}/{agreement['compared']}",
                 agreement["agreed"] >= TARGETS["cold_agreement_min"]))

    c = results["confidence"]
    gap = c["accuracy_most_confident_third"] - c["accuracy_least_confident_third"]
    rows.append((5, "Confidence gap (most- minus least-confident third)",
                 f">= {TARGETS['confidence_gap_min']:.2f}",
                 f"{c['accuracy_most_confident_third']:.3f} - "
                 f"{c['accuracy_least_confident_third']:.3f} = {gap:+.3f} "
                 f"(n={c['n_per_third']} per third)",
                 gap >= TARGETS["confidence_gap_min"] - 1e-9))
    return rows


def main():
    import pandas as pd

    for path in ("results.json", "labels.csv", "data/cold20_agreement.json"):
        if not Path(path).exists():
            print(f"No {path}. See the docstring for which step writes it.", file=sys.stderr)
            return 1

    results = json.loads(Path("results.json").read_text())
    counts = pd.read_csv("labels.csv")["label"].value_counts().to_dict()
    agreement = json.loads(Path("data/cold20_agreement.json").read_text())
    rows = evaluate(results, counts, agreement)

    print(f"Seed {results['seed']}, n_test {results['n_test']}, device {results['device']}\n")
    print("| # | Criterion | Target | Actual | Verdict |")
    print("|---|---|---|---|---|")
    for n, name, target, actual, met in rows:
        print(f"| {n} | {name} | {target} | {actual} | {'MET' if met else 'MISSED'} |")

    Path("criteria_results.json").write_text(json.dumps({
        "seed": results["seed"], "n_test": results["n_test"],
        "criteria": [{"number": n, "criterion": name, "target": target,
                      "actual": actual, "met": met}
                     for n, name, target, actual, met in rows],
    }, indent=2))
    print("\nWrote criteria_results.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
