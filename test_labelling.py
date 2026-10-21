#!/usr/bin/env python3
"""
Tests for the data pipeline: collect.py, labelling.py, and the notebook's split.

    python test_labelling.py

Separate from test.py, which checks the environment. Every post in here is a
made-up fixture for exercising code paths — none of it is, or goes near, the
dataset.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import collect
import labelling

ROOT = Path(__file__).resolve().parent
LABELS = ["a", "b", "c"]
WORDS = ("squat bench deadlift row press curl plank lunge sprint jog swim bike hike "
         "stretch protein sleep cardio grip core back chest legs arms rest").split()


def distinct_texts(n, seed=0):
    """n fixture posts that share no near-identical pair."""
    rng = random.Random(seed)
    return [" ".join(rng.sample(WORDS, 12)) + f" {i}" for i in range(n)]


def notebook_function(name):
    """Pull a function out of takemeter.ipynb so the test runs the real code."""
    nb = json.loads((ROOT / "takemeter.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and f"def {name}(" in source:
            # Keep only the definitions — the cell's trailing lines run on `df`.
            head = source.split("\ntrain_df, val_df, test_df")[0]
            namespace = {}
            exec(head, namespace)
            return namespace[name]
    raise LookupError(name)


def fixture_post(pid, title, body="", ap="", **flags):
    post = {"id": pid, "name": title, "body": body,
            "ap_id": ap or f"https://example.test/post/{pid}", "url": None,
            "published": "2026-01-01T00:00:00Z"}
    post.update(flags)
    return {"post": post}


class CollectProcessing(unittest.TestCase):
    def run_process(self, posts):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dump.json"
            path.write_text(json.dumps({"host": "example.test", "community": "c@x",
                                        "posts": posts}))
            return collect.process([path])

    def test_drops_removed_short_nsfw_and_duplicates(self):
        words = "one two three four five six seven eight nine"
        rows, dropped = self.run_process([
            fixture_post(1, "Title", words),
            fixture_post(2, "Title", words.upper() + "!!"),   # same words → dup text
            fixture_post(3, "Other", words, ap="https://example.test/post/1"),  # dup id
            fixture_post(4, "Too short"),
            fixture_post(5, "Gone", words, removed=True),
            fixture_post(6, "Gone", words, deleted=True),
            fixture_post(7, "Nope", words, nsfw=True),
        ])
        self.assertEqual([r["id"] for r in rows], ["example.test:1"])
        self.assertEqual(dropped, {"removed_or_deleted": 2, "nsfw": 1, "too_short": 1,
                                   "duplicate_id": 1, "duplicate_text": 1})

    def test_text_keeps_title_and_full_body(self):
        body = "line one\n\n\n\nline two   \nwith more words here to pass"
        rows, _ = self.run_process([fixture_post(1, "My question", body)])
        self.assertEqual(rows[0]["text"],
                         "My question\n\nline one\n\nline two\nwith more words here to pass")
        self.assertEqual(rows[0]["url"], "https://example.test/post/1")

    def test_title_not_repeated_when_body_starts_with_it(self):
        post = {"name": "Same start", "body": "Same start and then the rest"}
        self.assertEqual(collect.post_text(post), "Same start and then the rest")


class ColdReservation(unittest.TestCase):
    def test_deterministic_disjoint_and_order_independent(self):
        ids = [f"p{i}" for i in range(100)]
        cold, rest = labelling.reserve_cold(ids)
        cold2, rest2 = labelling.reserve_cold(list(reversed(ids)))
        self.assertEqual(cold, cold2)
        self.assertEqual(len(cold), 20)
        self.assertFalse(set(cold) & set(rest))
        self.assertEqual(set(cold) | set(rest), set(ids))
        self.assertEqual(rest, rest2)


class Validation(unittest.TestCase):
    def frame(self, n=200, labels=None, notes=None):
        labels = labels or [LABELS[i % 3] for i in range(n)]
        notes = notes or (["cold"] * 20 + ["pre-labelled by AI, human accepted"] * (n - 20))
        return pd.DataFrame({"text": distinct_texts(n),
                             "label": labels, "note": notes})

    def test_clean_frame_passes(self):
        problems, warnings = labelling.validate(self.frame(), LABELS)
        self.assertEqual(problems, [])
        self.assertEqual(warnings, [])

    def test_catches_each_problem(self):
        df = self.frame()
        df.loc[0, "label"] = "A"                       # wrong capitals
        df.loc[1, "text"] = df.loc[2, "text"]          # duplicate
        df.loc[3, "text"] = "  "                       # empty
        df.loc[4:30, "note"] = "pre-labelled"          # fewer than 20 cold
        problems, _ = labelling.validate(df, LABELS)
        joined = " | ".join(problems)
        for needle in ("not in", "duplicate", "empty text", "noted 'cold'"):
            self.assertIn(needle, joined)

    def test_catches_near_duplicates(self):
        df = self.frame()
        df.loc[5, "text"] = df.loc[9, "text"].upper() + " edited!"
        problems, _ = labelling.validate(df, LABELS)
        self.assertTrue(any("near-duplicate" in p for p in problems))

    def test_near_duplicates_finds_reposts_only(self):
        texts = ["What's your recent fitness victory? I want to know!",
                 "What's your recent fitness victory? We want to know!",
                 "Which lifting belt can you recommend for amateurs in Europe?"]
        self.assertEqual(labelling.near_duplicates(texts), [(0, 1)])

    def test_seventy_percent_cap(self):
        labels = ["a"] * 150 + ["b"] * 25 + ["c"] * 25
        problems, _ = labelling.validate(self.frame(labels=labels), LABELS)
        self.assertTrue(any("cap is 70%" in p for p in problems))

    def test_row_floor(self):
        problems, _ = labelling.validate(self.frame(n=120), LABELS)
        self.assertTrue(any("floor" in p for p in problems))


class NotebookSplit(unittest.TestCase):
    """The notebook's own split_dataset, read out of the .ipynb."""

    def setUp(self):
        self.split = notebook_function("split_dataset")
        n = 210
        self.df = pd.DataFrame({"text": [f"post {i}" for i in range(n)],
                                "label": [LABELS[i % 3] for i in range(n)]})

    def test_seventy_fifteen_fifteen_disjoint(self):
        train, val, test = self.split(self.df, 42)
        self.assertEqual(len(train) + len(val) + len(test), len(self.df))
        self.assertAlmostEqual(len(test) / len(self.df), 0.15, delta=0.01)
        self.assertAlmostEqual(len(val) / len(self.df), 0.15, delta=0.01)
        sets = [set(part["text"]) for part in (train, val, test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_same_seed_same_test_split(self):
        _, _, first = self.split(self.df, 42)
        _, _, second = self.split(self.df.sample(frac=1, random_state=0)
                                  .sort_index(), 42)
        self.assertEqual(list(first["text"]), list(second["text"]))

    def test_stratified(self):
        _, _, test = self.split(self.df, 42)
        counts = test["label"].value_counts()
        self.assertLessEqual(counts.max() - counts.min(), 1)


class RealArtifacts(unittest.TestCase):
    """Checks on the committed files themselves, once they exist."""

    @unittest.skipUnless((ROOT / "labels.csv").exists(), "labels.csv not built yet")
    def test_labels_csv_valid(self):
        df = pd.read_csv(ROOT / "labels.csv")
        labels = labelling.load_label_names(ROOT / "label_definitions.txt")
        problems, _ = labelling.validate(df, labels)
        self.assertEqual(problems, [])
        self.assertEqual(list(df.columns), ["text", "label", "note"])

    @unittest.skipUnless((ROOT / "data/labels_with_provenance.csv").exists(),
                         "labels not built yet")
    def test_provenance_is_honest(self):
        """Cold labels survive unchanged; no AI label is passed off as reviewed."""
        prov = pd.read_csv(ROOT / "data/labels_with_provenance.csv", dtype=str).fillna("")
        labels = pd.read_csv(ROOT / "labels.csv", dtype=str).fillna("")
        cold = pd.read_csv(ROOT / "data/cold20_to_label.csv", dtype=str).fillna("")
        self.assertEqual(list(prov["text"]), list(labels["text"]))
        self.assertEqual(list(prov["label"]), list(labels["label"]))

        human = prov[prov["label_source"] == "human_cold"].set_index("id")
        self.assertEqual(len(human), 20)
        for row in cold.itertuples():
            self.assertEqual(human.at[row.id, "label"], row.label.strip())
            self.assertEqual(human.at[row.id, "note"], "cold")

        ai = prov[prov["label_source"] == "ai_unreviewed"]
        self.assertTrue((ai["human_label"] == "").all())
        self.assertTrue((ai["note"] == labelling.UNREVIEWED_NOTE).all())
        self.assertTrue((ai["label"] == ai["ai_label"]).all())
        self.assertLessEqual(set(prov["label_source"]),
                             {"human_cold", "ai_unreviewed", "ai_human_reviewed"})

    @unittest.skipUnless((ROOT / "test_split.csv").exists() and (ROOT / "labels.csv").exists(),
                         "no test split yet")
    def test_test_split_is_the_seed_42_split_of_labels_csv(self):
        df = pd.read_csv(ROOT / "labels.csv")
        _, _, expected = notebook_function("split_dataset")(df, 42)
        saved = pd.read_csv(ROOT / "test_split.csv")
        self.assertEqual(list(saved["text"]), list(expected["text"]))

    @unittest.skipUnless((ROOT / "results.json").exists(), "no results.json yet")
    def test_results_consistent(self):
        r = json.loads((ROOT / "results.json").read_text())
        test = pd.read_csv(ROOT / "test_split.csv")
        self.assertEqual(r["n_test"], len(test))
        self.assertEqual(sum(v["support"] for v in r["per_label"].values()), len(test))
        self.assertEqual(sum(sum(row.values()) for row in r["confusion"].values()), len(test))
        for label, row in r["confusion"].items():
            self.assertEqual(sum(row.values()), r["per_label"][label]["support"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
