#!/usr/bin/env python3
"""
The labelling workflow. ← UNIT 5, MILESTONE 4

Four steps, in order. Each refuses to run until the one before it is done.

    python labelling.py prepare    # reserve the cold 20, write the files to label
    python labelling.py cold       # label the cold 20 by hand, no suggestions shown
    python labelling.py agree      # cold 20: human vs blind AI labels (criterion 4)
    python labelling.py review     # walk every AI pre-label: accept or correct
    python labelling.py status     # progress, corrections, projected balance
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
COLD_AI = Path("data/cold20_ai_labels.csv")
COLD_AGREEMENT = Path("data/cold20_agreement.json")

# The dataset is the two general fitness communities. running@lemmy.world was
# collected too, and is kept in data/raw, but left out: it is mostly link
# shares and race reports, a second community with its own style.
INCLUDED_SOURCES = ("fitness@lemmy.world", "fitness@lemmy.ml")

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


def near_duplicates(texts, threshold=0.85, prefix=600):
    """
    Pairs (i, j), i < j, whose words are near-identical.

    Exact-match dedup misses an edited repost or a cross-post with a different
    title, and either would put the same post on both sides of the split.
    Compares the first `prefix` characters of the lowercased words.
    """
    from difflib import SequenceMatcher

    keys = [re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()[:prefix] for t in texts]
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if not a or not b or abs(len(a) - len(b)) > 0.5 * max(len(a), len(b)):
                continue
            if SequenceMatcher(None, a, b).ratio() >= threshold:
                pairs.append((i, j))
    return pairs


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
    near = near_duplicates(text.tolist())
    if near:
        problems.append(f"{len(near)} near-duplicate pairs (rows {near[:3]}…) — "
                        f"reposts would leak across splits.")

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
    posts = posts[posts["source"].isin(INCLUDED_SOURCES)]
    cold_ids, rest_ids = reserve_cold(posts["id"].tolist())

    by_id = posts.set_index("id")
    cold = by_id.loc[cold_ids, ["url", "text"]].reset_index()
    cold["label"] = ""
    cold.to_csv(COLD, index=False)
    by_id.loc[rest_ids, ["source", "url", "text"]].reset_index().to_csv(TO_PRELABEL, index=False)

    print(f"Reserved {len(cold)} cold posts → {COLD}")
    print(f"{len(rest_ids)} posts for pre-labelling → {TO_PRELABEL}")
    print(f"\nLabel them with `python labelling.py cold`, or fill the `label`")
    print(f"column of {COLD} by hand, using one of:")
    print("  " + ", ".join(load_label_names()))


def cmd_cold(args, pd):
    """Label the cold 20 one at a time. No suggestion is shown, by design."""
    if not COLD.exists():
        raise SystemExit(f"No {COLD}. Run `python labelling.py prepare` first.")
    labels = load_label_names()
    df = pd.read_csv(COLD, dtype=str).fillna("")
    pending = [i for i in df.index if not df.at[i, "label"].strip()]
    print(f"{len(df) - len(pending)} of {len(df)} cold posts labelled. {len(pending)} to go.")
    print("Definitions are in label_definitions.txt and the README. No AI suggestion here.\n")
    menu = "  ".join(f"[{n}] {name}" for n, name in enumerate(labels, 1))

    for done, i in enumerate(pending, 1):
        print("─" * 72)
        print(f"cold {done}/{len(pending)}   {df.at[i, 'url']}\n")
        print(df.at[i, "text"])
        print(f"\n{menu}")
        while True:
            choice = input("number = label · s = skip · q = quit > ").strip().lower()
            if choice == "q":
                df.to_csv(COLD, index=False)
                print("Saved. Resume with `python labelling.py cold`.")
                return
            if choice == "s":
                break
            if choice.isdigit() and 1 <= int(choice) <= len(labels):
                df.at[i, "label"] = labels[int(choice) - 1]
                df.to_csv(COLD, index=False)
                break
            print("  ?")

    left = int((df["label"].str.strip() == "").sum())
    print(f"\nDone for now. {left} cold posts still unlabelled.")


def cmd_agree(args, pd):
    """Criterion 4: the human cold labels against the blind AI labels."""
    import json
    for path in (COLD, COLD_AI):
        if not path.exists():
            raise SystemExit(f"No {path}.")
    human = pd.read_csv(COLD, dtype=str).fillna("")
    ai = pd.read_csv(COLD_AI, dtype=str).fillna("")
    both = human.merge(ai, on="id", how="inner")
    if len(both) != len(human) or (both["label"].str.strip() == "").any():
        raise SystemExit("Every cold post needs a human label and an AI label.")
    both["label"] = both["label"].str.strip()
    agreed = int((both["label"] == both["ai_label"]).sum())
    labels = load_label_names()
    matrix = {h: {a: int(((both["label"] == h) & (both["ai_label"] == a)).sum())
                  for a in labels} for h in labels}
    disagreements = [
        {"id": r.id, "url": r.url, "human": r.label, "ai": r.ai_label,
         "ai_reason": r.ai_reason, "text": r.text}
        for r in both.itertuples() if r.label != r.ai_label
    ]
    COLD_AGREEMENT.write_text(json.dumps({
        "compared": len(both), "agreed": agreed, "agreement_rate": agreed / len(both),
        "target": 15, "met": agreed >= 15,
        "human_counts": both["label"].value_counts().to_dict(),
        "ai_counts": both["ai_label"].value_counts().to_dict(),
        "matrix_human_rows_ai_columns": matrix,
        "disagreements": disagreements,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Agreed on {agreed} of {len(both)} ({agreed / len(both):.0%}). "
          f"Criterion 4 target 15/20: {'MET' if agreed >= 15 else 'MISSED'}\n")
    print("| human \\ AI | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    for h in labels:
        print(f"| **{h}** | " + " | ".join(str(matrix[h][a]) for a in labels) + " |")
    for d in disagreements:
        print(f"\n[{d['id']}] human={d['human']}  ai={d['ai']}  ({d['ai_reason']})")
        print("  " + normalise(d["text"])[:220])
    print(f"\nWrote {COLD_AGREEMENT}.")


def cmd_review(args, pd):
    if not PRELABELS.exists():
        raise SystemExit(f"No {PRELABELS} yet — the AI pre-labels go there.")
    labels = load_label_names()
    df = pd.read_csv(PRELABELS, dtype=str).fillna("")
    for column in ("human_label", "note"):
        if column not in df.columns:
            df[column] = ""

    pending = [i for i in df.index if not df.at[i, "human_label"]]
    if args.borderline:
        pending = [i for i in pending if df.at[i, "ai_borderline"] == "True"]
    print(f"{int((df['human_label'] != '').sum())} of {len(df)} reviewed. "
          f"{len(pending)} to go{' (borderline only)' if args.borderline else ''}.\n")
    menu = "  ".join(f"[{n}] {name}" for n, name in enumerate(labels, 1))

    history = []
    step = 0
    while step < len(pending):
        i = pending[step]
        row = df.loc[i]
        print("─" * 72)
        flag = "   ⚠ BORDERLINE — read it closely" if row.get("ai_borderline") == "True" else ""
        print(f"{step + 1}/{len(pending)}   {row['id']}   {row.get('url', '')}{flag}\n")
        print(row["text"])
        print(f"\nAI suggests: {row['ai_label']}   ({row.get('ai_reason', '')})")
        print(menu)
        choice = input("Enter = accept · number = change · u = undo · s = skip · q = quit > ").strip().lower()
        if choice == "q":
            break
        if choice == "u":
            if history:
                step = history.pop()
                df.at[pending[step], "human_label"] = ""
                df.at[pending[step], "note"] = ""
                df.to_csv(PRELABELS, index=False)
            else:
                print("  nothing to undo")
            continue
        if choice == "s":
            step += 1
            continue
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
        history.append(step)
        step += 1

    df.to_csv(PRELABELS, index=False)
    reviewed = df[df["human_label"] != ""]
    changed = int((reviewed["human_label"] != reviewed["ai_label"]).sum())
    print(f"\nSaved. {len(reviewed)} reviewed, {changed} corrected so far.")


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


def cmd_status(args, pd):
    """Where the review stands, and what the label balance would be if built now."""
    pre = pd.read_csv(PRELABELS, dtype=str).fillna("")
    cold = pd.read_csv(COLD, dtype=str).fillna("")
    reviewed = pre[pre["human_label"] != ""]
    left = pre[pre["human_label"] == ""]
    print(f"cold labelled     {int((cold['label'].str.strip() != '').sum())} / {len(cold)}")
    print(f"pre-labels        {len(reviewed)} reviewed, {len(left)} to go "
          f"({int((left['ai_borderline'] == 'True').sum())} of those borderline)")
    if len(reviewed):
        changed = int((reviewed["human_label"] != reviewed["ai_label"]).sum())
        print(f"                  {changed} corrected ({changed / len(reviewed):.0%} of reviewed)")
    # Projected balance: human label where there is one, AI suggestion otherwise.
    projected = pd.concat([cold["label"].str.strip(),
                           pre["human_label"].where(pre["human_label"] != "", pre["ai_label"])])
    counts = projected.value_counts()
    print("\nprojected balance (criterion 3: largest <= 65%, smallest >= 15%)")
    for name, count in counts.items():
        print(f"  {name:<16} {count:>4}  {count / len(projected):.1%}")


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
    sub.add_parser("cold")
    sub.add_parser("agree")
    r = sub.add_parser("review")
    r.add_argument("--borderline", action="store_true",
                   help="only the posts the AI flagged as close calls")
    b = sub.add_parser("build")
    b.add_argument("--allow-unreviewed", action="store_true",
                   help="drop unreviewed pre-labels instead of refusing")
    sub.add_parser("check")
    sub.add_parser("status")
    args = parser.parse_args()

    import pandas as pd
    handler = {"prepare": cmd_prepare, "cold": cmd_cold, "agree": cmd_agree, "review": cmd_review,
               "build": cmd_build, "check": cmd_check, "status": cmd_status}[args.command]
    return handler(args, pd) or 0


if __name__ == "__main__":
    sys.exit(main())
