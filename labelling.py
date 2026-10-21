#!/usr/bin/env python3
"""
The labelling workflow. ← UNIT 5, MILESTONE 4

Four steps, in order. Each refuses to run until the one before it is done.

    python labelling.py prepare    # reserve the cold 20, write the files to label
    python labelling.py review     # walk every AI pre-label: accept or correct
    python labelling.py build      # merge cold + reviewed into labels.csv
    python labelling.py check      # validate labels.csv against the brief

The label set is read from `label_definitions.txt`, the same file baseline.py
reads, so there is one place the labels are spelled.

**The cold 20.** `prepare` shuffles the processed posts with a fixed seed and
reserves the first 20 for labelling by hand, with no model suggestion anywhere
near them. They go to `data/cold20_to_label.csv` with an empty `label` column.
The rest go to `data/to_prelabel.csv`, which is what the AI pre-labels.

**The review.** `review` shows each pre-labelled post with the suggested label
and its one-line reason. Enter accepts, a number picks a different label, `s`
skips, `q` saves and quits. Progress is saved after every decision, so it can
be stopped and resumed. The note column records what happened to each one.

You can also review in a spreadsheet: fill `human_label` in
`data/prelabels.csv` directly. `build` only uses rows that have one.
"""

import argparse
import random
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass


SEED = 42
N_COLD = 20

DEFINITIONS = Path("label_definitions.txt")
PROCESSED = Path("data/posts_processed.csv")
COLD = Path("data/cold20_to_label.csv")
TO_PRELABEL = Path("data/to_prelabel.csv")
PRELABELS = Path("data/prelabels.csv")
LABELS_CSV = Path("labels.csv")
PROVENANCE = Path("data/labels_with_provenance.csv")

COLD_NOTE = "cold"
MAX_SHARE = 0.70
MIN_ROWS = 150
TARGET_ROWS = 200


def load_label_names(path=DEFINITIONS):
    """The labels, in file order, from `label: definition` lines."""
    if not Path(path).exists():
        raise SystemExit(f"No {path}. Put your labels in it as `label: definition`.")
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and ":" in line:
            labels.append(line.split(":", 1)[0].strip())
    if len(labels) < 2:
        raise SystemExit(f"{path} defines {len(labels)} labels; a classifier needs at least 2.")
    return labels


def normalise(text):
    """Match posts on their words, not on stray whitespace."""
    return re.sub(r"\s+", " ", str(text)).strip()


def reserve_cold(ids, n=N_COLD, seed=SEED):
    """A fixed shuffle; the first n are the cold set. Returns (cold_ids, rest_ids)."""
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return order[:n], order[n:]


# ── validation ──────────────────────────────────────────────────────────────

def validate(df, labels):
    """
    Everything the brief and the notebook check, plus the labelling rules.

    Returns (problems, warnings). Problems block a build; warnings are for the README.
    """
    problems, warnings = [], []

    for column in ("text", "label", "note"):
        if column not in df.columns:
            problems.append(f"No '{column}' column. Found: {', '.join(df.columns)}")
    if problems:
        return problems, warnings

    text = df["text"].fillna("").astype(str)
    if (text.str.strip() == "").any():
        problems.append(f"{int((text.str.strip() == '').sum())} rows have empty text.")

    label = df["label"].fillna("").astype(str).str.strip()
    if (label == "").any():
        problems.append(f"{int((label == '').sum())} rows have no label.")
    unknown = sorted(set(label[label != ""]) - set(labels))
    if unknown:
        problems.append(f"Labels not in {DEFINITIONS}: {unknown}")
    unused = sorted(set(labels) - set(label))
    if unused:
        problems.append(f"Labels that never appear: {unused}")

    duplicates = int(text.map(normalise).duplicated().sum())
    if duplicates:
        problems.append(f"{duplicates} duplicate posts — they would leak across splits.")

    total = len(df)
    if total:
        for name, count in label.value_counts().items():
            if count / total > MAX_SHARE:
                problems.append(f"'{name}' is {count / total:.0%} of the data; the cap is 70%.")
            if count < 20:
                warnings.append(f"Only {count} examples of '{name}' — its test F1 will be jumpy.")

    if total < MIN_ROWS:
        problems.append(f"{total} rows; the submission floor is {MIN_ROWS}.")
    elif total < TARGET_ROWS:
        warnings.append(f"{total} rows — under the {TARGET_ROWS} the brief asks for.")

    cold = int((df["note"].fillna("").astype(str).str.strip() == COLD_NOTE).sum())
    if cold < N_COLD:
        problems.append(f"{cold} rows noted '{COLD_NOTE}'; the brief asks for {N_COLD}.")

    return problems, warnings


# ── steps ───────────────────────────────────────────────────────────────────

def cmd_prepare(args, pd):
    if not PROCESSED.exists():
        raise SystemExit(f"No {PROCESSED}. Run `python collect.py` first.")
    if COLD.exists() and not args.force:
        raise SystemExit(f"{COLD} already exists. The cold set is reserved once; "
                         f"--force only if nothing has been labelled yet.")

    posts = pd.read_csv(PROCESSED)
    cold_ids, rest_ids = reserve_cold(posts["id"].tolist())

    by_id = posts.set_index("id")
    cold = by_id.loc[cold_ids, ["url", "text"]].reset_index()
    cold["label"] = ""
    cold.to_csv(COLD, index=False)
    by_id.loc[rest_ids, ["source", "url", "text"]].reset_index().to_csv(TO_PRELABEL, index=False)

    print(f"Reserved {len(cold)} cold posts → {COLD}")
    print(f"{len(rest_ids)} posts for pre-labelling → {TO_PRELABEL}")
    print(f"\nFill the `label` column of {COLD} by hand, using one of:")
    print("  " + ", ".join(load_label_names()))


def cmd_review(args, pd):
    if not PRELABELS.exists():
        raise SystemExit(f"No {PRELABELS} yet — the AI pre-labels go there.")
    labels = load_label_names()
    df = pd.read_csv(PRELABELS, dtype=str).fillna("")
    for column in ("human_label", "note"):
        if column not in df.columns:
            df[column] = ""

    pending = [i for i in df.index if not df.at[i, "human_label"]]
    print(f"{len(df) - len(pending)} of {len(df)} reviewed. {len(pending)} to go.\n")
    menu = "  ".join(f"[{n}] {name}" for n, name in enumerate(labels, 1))

    for done, i in enumerate(pending, 1):
        row = df.loc[i]
        print("─" * 72)
        print(f"{done}/{len(pending)}   {row['id']}   {row.get('url', '')}\n")
        print(row["text"])
        print(f"\nAI suggests: {row['ai_label']}   ({row.get('ai_reason', '')})")
        print(menu)
        while True:
            choice = input("Enter = accept · number = change · s = skip · q = quit > ").strip().lower()
            if choice == "q":
                df.to_csv(PRELABELS, index=False)
                print(f"Saved. Resume with `python labelling.py review`.")
                return
            if choice == "s":
                break
            if choice == "":
                picked = row["ai_label"]
            elif choice.isdigit() and 1 <= int(choice) <= len(labels):
                picked = labels[int(choice) - 1]
            else:
                print("  ?")
                continue
            df.at[i, "human_label"] = picked
            df.at[i, "note"] = ("pre-labelled by AI, human accepted"
                                if picked == row["ai_label"]
                                else f"pre-labelled by AI as {row['ai_label']}, human corrected")
            df.to_csv(PRELABELS, index=False)
            break

    left = int((df["human_label"] == "").sum())
    print(f"\nDone for now. {left} still unreviewed.")


def cmd_build(args, pd):
    labels = load_label_names()
    for path in (COLD, PRELABELS):
        if not path.exists():
            raise SystemExit(f"No {path}.")

    cold = pd.read_csv(COLD, dtype=str).fillna("")
    if (cold["label"].str.strip() == "").any():
        n = int((cold["label"].str.strip() == "").sum())
        raise SystemExit(f"{n} of the cold posts in {COLD} still have no label.")
    cold = cold.assign(label=cold["label"].str.strip(), note=COLD_NOTE)

    pre = pd.read_csv(PRELABELS, dtype=str).fillna("")
    unreviewed = int((pre["human_label"].str.strip() == "").sum())
    if unreviewed and not args.allow_unreviewed:
        raise SystemExit(f"{unreviewed} pre-labels have no human_label. Every AI label "
                         f"is read by a person before it trains anything — finish "
                         f"`python labelling.py review` first.")
    pre = pre[pre["human_label"].str.strip() != ""]
    pre = pre.assign(label=pre["human_label"].str.strip())

    columns = ["id", "url", "text", "label", "note"]
    merged = pd.concat([cold[columns], pre[columns]], ignore_index=True)

    problems, warnings = validate(merged, labels)
    for w in warnings:
        print(f"[look at]  {w}")
    for p in problems:
        print(f"[FIX ME]   {p}")
    if problems:
        raise SystemExit("\nNot written. Fix the [FIX ME] lines.")

    merged.to_csv(PROVENANCE, index=False)
    merged[["text", "label", "note"]].to_csv(LABELS_CSV, index=False)
    print(f"\nWrote {LABELS_CSV} ({len(merged)} rows) and {PROVENANCE}.")
    print(merged["label"].value_counts().to_string())


def cmd_check(args, pd):
    labels = load_label_names()
    df = pd.read_csv(LABELS_CSV)
    problems, warnings = validate(df, labels)
    for w in warnings:
        print(f"[look at]  {w}")
    for p in problems:
        print(f"[FIX ME]   {p}")
    print(f"\n{len(df)} rows, {'OK' if not problems else 'NOT OK'}.")
    counts = df["label"].value_counts()
    print((pd.DataFrame({"count": counts, "share": (counts / len(df)).round(3)})).to_string())
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--force", action="store_true")
    sub.add_parser("review")
    b = sub.add_parser("build")
    b.add_argument("--allow-unreviewed", action="store_true",
                   help="drop unreviewed pre-labels instead of refusing")
    sub.add_parser("check")
    args = parser.parse_args()

    import pandas as pd
    handler = {"prepare": cmd_prepare, "review": cmd_review,
               "build": cmd_build, "check": cmd_check}[args.command]
    return handler(args, pd) or 0


if __name__ == "__main__":
    sys.exit(main())
