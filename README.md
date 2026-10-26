# TakeMeter

> ### 👋 Start here
>
> **New to this repo? Read [RUNNING.md](RUNNING.md) first** — setup, the
> notebook, the baseline, and what to do when something breaks.
>
> Once `python test.py` passes:
>
> ```bash
> head -5 data/practice_labels.csv     # the shape your labels.csv needs
> ```
>
> Then open `takemeter.ipynb` **in this folder** — in VS Code, or with
> `jupyter notebook` if you prefer. Pick the kernel: the `.venv` inside this
> project. Run section 1, which reports the hardware you'll be training on.
> Everything else waits until you have data.
>
> Nothing to upload, nothing to connect, no accounts and no keys. The notebook
> runs on your machine and writes next to your code.
>
> **The rest of this file is your submission.** Fill it in as you go.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     HOW TO USE THIS FILE

     Unit 5 asks for the first five sections. Unit 6 adds the five below them.

     Everything is pasted as TEXT. No screenshots, no images.

     ⚠️ The confusion matrix especially. The notebook prints one as a markdown
     table, ready to copy. A screenshot of a matrix earns nothing. Paste the
     table.
     ───────────────────────────────────────────────────────────────────────── -->

<!-- ═══════════════════════ UNIT 5 — THE BUILD ═══════════════════════ -->

## What This Does

TakeMeter sorts posts from public Lemmy fitness communities (`fitness@lemmy.world`
and `fitness@lemmy.ml`, the closest open equivalent of r/Fitness) by what the
writer is doing: **asking** the community something, **reporting** on their own
fitness journey, or **passing on** information. It is a fine-tuned
`distilbert-base-uncased` classifier trained on 371 real posts. A sorter like this
would let a moderator or reader pull out the progress posts, which are rare and
easily buried under questions and link shares.

> ⚠️ **Read before trusting any number here:** 351 of the 371 training labels
> were produced by an AI and **were not individually reviewed by a human**.
> Only the 20 "cold" posts were labelled by hand. See
> [How I Used AI](#how-i-used-ai).

---

## Label Taxonomy

The provisional plan had four labels: `question`, `advice`, `progress` and
`experience`. Reading a 40-post sample changed that. Advice almost never
appears as a *post* on these communities (it lives in comments). `progress`
and `experience` were each about 8% of posts and hard to tell apart. And about
20% of posts were article or study links that fit none of the four. The final
taxonomy has three labels.

### `question`

**Definition:** the post's main purpose is to get an answer, a recommendation,
or feedback from the community, including "rate my routine" posts and polls.

**Example 1:** (cold, labelled by hand) [sopuli.xyz/post/50986271](https://sopuli.xyz/post/50986271)
> What fitness tracker do you use that supports easy data export? I'm thinking about getting a fitness tracker to use together with endurain. Garmin trackers apparently support automatic synchronization with endurain, but they are very expensive. What fitness tracker do you use that supports simple data export in a common format?

**Example 2:** (cold, labelled by hand) [lemmy.world/post/12627708](https://lemmy.world/post/12627708)
> home pull up help. I read that resistance bands on me can help me get going on pull up bars. Is there a specific kind or style or size I should get for these? Thanks all.

### `personal_update`

**Definition:** the writer reports on their own fitness journey (an
achievement, milestone, setback, a routine they are trying, or how something
felt), and the post is not mainly asking for anything.

**Example 1:** (cold, labelled by hand) [lemmy.world/post/856720](https://lemmy.world/post/856720)
> Hiked 60+ miles in June. Just got back into hiking after an extended leave. I managed to get over 60 miles in June which isn't too bad for me. […] I am glad I did because I am more motivated and my eating habits are getting better. My goal for next month is 100 miles.

**Example 2:** (AI-labelled, not reviewed) [lemmy.world/post/106954](https://lemmy.world/post/106954)
> NSV - Lifted more than my bodyweight for the first time. Oh man I feel high on endorphins. I (27F) got into heavy lifting at the start of April […] I PR'd my squat (65kg) AND my deadlift (75kg) today.

### `shared_info`

**Definition:** the post passes on general fitness information not centred on
the writer's own story: a linked article or study, tips, a guide, a product
write-up, or a promotion.

**Example 1:** (cold, labelled by hand) [lemmy.world/post/28083837](https://lemmy.world/post/28083837)
> MSU research: Eating brown rice increases exposure to arsenic compared to white rice

**Example 2:** (cold, labelled by hand) [lemmy.world/post/1264933](https://lemmy.world/post/1264933)
> Jeff Nippard's "The Ugly Truth About Getting Shredded (Science Explained)". Thought this was mindblowing. […] he references studies and talks about things men might experience at 5% body fat that I never expected, like extremely low testosterone.

### The hardest boundary

**Which two labels:** `question` vs `personal_update`

**The decision rule I used every time:** *delete the questions and see if the
post still stands.* If what's left is a complete story worth posting, it's
`personal_update`. If removing the questions leaves only context for an ask,
it's `question`. A tacked-on "anyone else?" doesn't make a post a question.

Two more rules cover the rest:
- **`personal_update` vs `shared_info`: who is the subject?** If it's the
  writer's own experience, it's `personal_update`. If it's generalisable
  guidance or someone else's content, it's `shared_info`, even when framed as
  "what worked for me".
- **Tie-break:** pick the label the post spends most of its words on, and only
  after the two rules above.

The cold-20 comparison shows the boundary really is the hard one: two of the
three disagreements between my labels and the AI's blind labels fall exactly
here (see hard cases 1 and 2).

---

## The Dataset

**Where the posts came from:** the public Lemmy API (`/api/v3/post/list`),
from `fitness@lemmy.world` (311 posts) and `fitness@lemmy.ml` (99), collected
on 2026-10-20 by `collect.py`. Posts date from 2020-05 to 2026-10.

Reddit was the first choice, but its anonymous API returns HTTP 403 and its
`robots.txt` disallows every path. Getting Reddit posts would have meant
working around that, so I didn't. The Lemmy instances' `robots.txt` asks for a
60-second crawl delay, and the collector waits that long between requests.
Nothing logs in.

`running@lemmy.world` (533 posts) was also collected and is kept in
`data/raw/`, but it isn't in the dataset. It is mostly link shares and race
reports, a second community with its own style.

From the 410 fitness posts:
- 19 were dropped as too short (under 8 words, usually an image or link with a
  bare title).
- 17 were exact-text duplicates.
- 3 more were near-duplicate reposts found with a similarity check: an edited
  copy, a cross-post with a different title, and a templated weekly thread
  near-identical to one of the cold 20. They are logged with reasons in
  `data/excluded_near_duplicates.csv`.

That leaves **371 posts**. Every post keeps its source URL in
`data/labels_with_provenance.csv`. The raw API responses are in `data/raw/`.

**How I labelled them:**
1. **20 cold posts, by hand, with no AI suggestion.** They were picked by a
   seeded shuffle (`labelling.py prepare`, seed 42) and labelled with
   `labelling.py cold`. The tool shows no suggestion. These are the rows noted
   `cold`.
2. **The other 351 were labelled by an AI (Claude) and NOT reviewed by me.**
   The AI read every post and gave each a label, a one-line reason, and a
   "borderline" flag on 40 close calls (`data/prelabels.csv`). A review tool
   was built (`labelling.py review`), but I chose not to review the labels
   individually, to finish faster. Each of these rows is noted
   `AI-labelled, NOT reviewed by a human` in `labels.csv`, and its
   `human_label` is empty in the provenance file. A test
   (`test_provenance_is_honest`) fails if any AI label is ever recorded as
   reviewed.
3. **A consistency check on the cold 20.** After my cold labels were
   committed, the AI labelled the same 20 posts blind. We agreed on 17 of 20
   (85%; `data/cold20_agreement.json`). That is the only evidence of how well
   the AI's labels match mine, and it covers 20 posts, not 351.

**Counts per label:**

| Label | Count | Share | Of which human-labelled |
|---|---|---|---|
| `question` | 216 | 58.2% | 14 |
| `shared_info` | 111 | 29.9% | 3 |
| `personal_update` | 44 | 11.9% | 3 |
| **Total** | 371 | 100% | 20 |

**Three hard cases**

These are the three cold posts where my label and the AI's blind label
disagreed. In each case I explain why it was hard and what the rules point to.

**1.** [lemmy.world/post/27935404](https://lemmy.world/post/27935404)
> *The post:* "I like going to gym but I feel like I am making no progress. […] I started with 7,5 kg dumbbell presses and now I am at 12,5 kg. […] Dunno what I am doing wrong? Doing Push Pull Leg 6 days a week […] I feel so bad. Im doing everything I can"
>
> *Could have been:* `question` or `personal_update`
>
> *I chose:* `personal_update` (the AI chose `question`). Applying rule 1 and
> deleting "Dunno what I am doing wrong?" leaves a complete post: progress
> numbers, a routine, and how the writer feels about it. The post ends on a
> feeling, not a request. The AI read the numbers as context for the ask. It
> is the clearest example of why this boundary is the hardest.

**2.** [sopuli.xyz/post/17740204](https://sopuli.xyz/post/17740204)
> *The post:* "Supersetting with mobility. Hi y'all. So I'm closing in on a full year at the gym. […] I've got the best/most fun body I've had in at least 15 years. […] So let me have your favorite mobility inserts into super sets!"
>
> *Could have been:* `question` or `personal_update`
>
> *I chose:* `personal_update` (the AI chose `question`). Most of the post is a
> reflection on a year of training: what has worked, the classes, enjoying
> moving a "new" body. The request for superset ideas comes at the end. Rule
> 1 leaves a complete story, and the tie-break (most words) points the same way.

**3.** [lemmy.world/post/33704050](https://lemmy.world/post/33704050)
> *The post:* "What Dr Mike Gets WRONG About Kettlebell Swings… AGAIN!"
>
> *Could have been:* `shared_info` or `question`
>
> *I chose:* `question` (the AI chose `shared_info`). It is a title-only post
> with no body. It reads like a video headline, which would be `shared_info`,
> but there's no link text to confirm it, and it can be read as inviting a
> discussion. Rule 2 doesn't settle a post this short. It shows a gap in
> my definitions: a bare headline with no body is ambiguous under all three.

---

## The Training Run

**Base model:** `distilbert-base-uncased` (the starter default)

**Settings:** 3 epochs, learning rate 2e-5, batch size 16, max length 128,
seed 42. These are all starter defaults. Device: **Apple Silicon GPU (MPS)**,
torch 2.14.1, transformers 4.57.6. Training took 41 seconds (51 optimizer
steps).

**Anything I changed from the defaults, and why:** nothing in training. In
the notebook I only set `LABELS`, `NAME` and `EMAIL`. Sections 1–5 were run
headless with `run_notebook.py` in this project's `.venv`, which executes the
same cells in order and stops at the Unit 6 marker. The outputs are saved in
`takemeter.ipynb`.

**Split sizes:** train 259 · val 56 · test 56 (stratified 70/15/15, seed 42)

| Label | train | val | test |
|---|---|---|---|
| `question` | 151 | 33 | 32 |
| `personal_update` | 31 | 6 | **7** |
| `shared_info` | 77 | 17 | 17 |

`personal_update` has **only 7 posts in the test split**, below the roughly 8
the notebook warns about, so its F1 rests on 7 posts and will move a lot
between seeds.

**Validation during training** (from the notebook's training log; the test
split was not used until the final evaluation):

| Epoch | Train loss | Val loss | Val accuracy | Val macro F1 |
|---|---|---|---|---|
| 1 | 1.000 | 0.819 | 0.589 | 0.247 |
| 2 | 0.784 | 0.725 | 0.696 | 0.439 |
| 3 | 0.692 | 0.686 | 0.786 | 0.544 |

Validation loss was still falling when training stopped.

---

## Results (Unit 5, seed 42, held-out test split)

From `results.json`. The test split (`test_split.csv`, 56 posts) was used once,
after training, with no changes made afterwards.

| Measure | Value |
|---|---|
| Accuracy | 0.679 |
| Macro F1 | 0.434 |

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `question` | 0.640 | 1.000 | 0.780 | 32 |
| `personal_update` | 0.000 | 0.000 | 0.000 | 7 |
| `shared_info` | 1.000 | 0.353 | 0.522 | 17 |

**Confusion matrix** (rows are true labels, columns are predictions):

| true \ predicted | question | personal_update | shared_info |
|---|---|---|---|
| **question** | 32 | 0 | 0 |
| **personal_update** | 7 | 0 | 0 |
| **shared_info** | 11 | 0 | 6 |

The model has mostly learned "say `question`". It called every
`personal_update` post `question`, and 11 of 17 `shared_info` posts too. It
never predicted `personal_update` at all. The 6 `shared_info` posts it did
catch were all correct (precision 1.000), so it has started to pick up
headline-style link posts, but only the clearest ones.

Confidence does carry information. The most-confident third of test posts
was 94.4% accurate, against 44.4% for the least-confident third (18 posts
each). Mean confidence was 0.70 when right and 0.58 when wrong.

**Acceptance criteria** (`criteria.md`, filed before training; computed by
`criteria_check.py` and saved in `criteria_results.json`):

| # | Criterion | Target | Actual | Verdict |
|---|---|---|---|---|
| 1 | Test accuracy | ≥ 0.75 | 0.679 | **MISSED** |
| 2 | Per-label F1 | `question` ≥ 0.80, `shared_info` ≥ 0.60, `personal_update` ≥ 0.50 | 0.780, 0.522, 0.000 | **MISSED** (all three) |
| 3 | Balance in `labels.csv` | largest ≤ 65%, smallest ≥ 15% | 58.2%, 11.9% | **MISSED** (smallest) |
| 4 | Cold-20 agreement, human vs blind AI | ≥ 15/20 | 17/20 | **MET** |
| 5 | Confidence gap, most- minus least-confident third | ≥ 0.15 | 0.944 − 0.444 = 0.500 | **MET** |

Two of five met. The criteria were not changed after the results came in.
Criterion 5's target was set back from 0.10 to 0.15 *before* training, and
the git history shows the order. These are one seed, and Unit 6 tests each
target across three.

A caveat on criterion 5. Part of the gap comes from the model being
confidently right on obvious questions, which make up most of the test set.
The gap shows that confidence would work for routing posts, not that the
model is good.

---

## Limitations

- **Most labels are unreviewed AI labels.** 351 of 371 labels (95%) were never
  checked by a person. The cold-20 agreement (17/20) suggests they are mostly
  in line with my reading, but it is 20 posts. The disagreements cluster on
  the `question`/`personal_update` boundary, which is also the boundary the
  model fails on. The model may have learned the AI's reading of that
  boundary rather than mine, or the labels may simply be noisy there, and I
  can't tell the two apart without reviewing them.
- **Undertrained.** Validation loss was still dropping at epoch 3, after only
  51 steps. The starter defaults were kept, as the brief asks, so this is
  recorded as a finding for Unit 6 rather than fixed.
- **Too few `personal_update` posts.** 44 in total, 31 in training and 7 in
  test, which is under the 15% balance floor I set myself. Its F1 of 0.000
  rests on 7 posts.
- **One seed.** Every number above comes from the seed-42 split of 56 test
  posts. A single post moves accuracy by about 1.8 points.
- **One small community.** Lemmy fitness communities are small and skew
  towards FOSS and privacy-minded users (many tracker and app questions).
  Results may not transfer to r/Fitness.
- **Title-only link posts are ambiguous.** See hard case 3. A bare headline
  with no body doesn't settle cleanly under any of the definitions.

---

## Reproducing this

```bash
source .venv/bin/activate
python test.py                         # environment check
python collect.py --process-only       # rebuild data/posts_processed.csv from data/raw
                                       # (python collect.py re-fetches; ~20 min at the 60 s crawl delay)
python labelling.py agree              # cold-20 agreement → data/cold20_agreement.json
python labelling.py build --include-unreviewed   # labels.csv + data/labels_with_provenance.csv
python labelling.py check              # validation against the brief
pip install nbclient ipykernel && python -m ipykernel install --user --name takemeter-venv
python run_notebook.py                 # notebook sections 1–5 → results.json, test_split.csv
python criteria_check.py               # five criteria → criteria_results.json
python test_labelling.py               # pipeline and artifact tests
```

Or open `takemeter.ipynb` in VS Code with the `.venv` kernel and run sections
1–5. A re-fetch with `collect.py` returns whatever the communities hold that
day, so for an exact reproduction use the committed `data/raw/`. Same seed on
different hardware can give slightly different numbers (see notebook section 1).

| File | What it is |
|---|---|
| `collect.py` | Polite Lemmy API collector, and the processing and dedup step |
| `labelling.py` | Cold reservation, cold labelling, agreement, review, build, check |
| `criteria_check.py` | Scores the five criteria from the run's files |
| `run_notebook.py` | Runs notebook sections 1–5 headless in the `.venv` |
| `test_labelling.py` | Tests for all of the above, plus checks on the committed artifacts |
| `data/raw/*.json` | Raw API responses as fetched |
| `data/posts_processed.csv` | 869 usable posts, all three communities |
| `data/cold20_to_label.csv` | The 20 cold posts and my labels |
| `data/cold20_ai_labels.csv` | The AI's blind labels for the same 20 |
| `data/prelabels.csv` | AI labels, reasons and borderline flags for the 351 (`human_label` empty) |
| `data/labels_with_provenance.csv` | `labels.csv` plus id, URL, label source, AI label and human label |
| `data/excluded_near_duplicates.csv` | The 3 reposts removed, and why |

---

## How I Used AI

I used Claude (Claude Code) as an engineering assistant throughout: it wrote
the collector, the labelling and review tools, the tests and this write-up,
and ran the notebook.

**Moment 1: the taxonomy**

- *What I asked for:* a final taxonomy built from my four provisional labels
  (`question`, `advice`, `progress`, `experience`), based on what is actually
  in a sample of real posts.
- *What came back:* after reading 40 real posts, it reported that `advice`
  barely exists as a post on these communities, that `progress` and
  `experience` were each about 8% with a fuzzy line between them, and that
  about 20% of posts (article and study links) fit none of my four labels.
  It proposed `question` / `personal_update` / `shared_info` with three
  decision rules.
- *What I changed:* I chose the 3-label version over the 4-label alternatives
  it offered, which meant dropping `advice` and merging `progress` and
  `experience`.

**Moment 2: criterion 5**

- *What I asked for:* help defining five measurable acceptance criteria.
- *What came back:* five criteria. The confidence criterion asked for a
  0.15 gap between the most- and least-confident thirds.
- *What I changed:* I first lowered it to 0.10, then set it back to 0.15. Both
  changes were committed before any model was trained.

**Pre-labelling disclosure:**

- **Human-labelled:** 20 posts (the `cold` rows), labelled by me with no AI
  suggestion shown, and committed before the AI saw them.
- **AI-labelled, NOT reviewed:** 351 posts. Claude read each post and
  assigned a label with a one-line reason. **I did not review or correct
  these labels individually.** The brief's allowed workflow is pre-labelling
  followed by reading and correcting every label. I did not do the reading
  and correcting step, and that requirement is **not met**. The labels are
  marked `AI-labelled, NOT reviewed by a human` in `labels.csv`.
- **Agreement check:** on the cold 20, the AI's blind labels matched mine on
  17 of 20.

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy | 0.893 | 0.679 | -0.214 |
| Macro F1 | 0.878 | 0.434 | -0.443 |
| F1 — `personal_update` | 0.857 | 0.000 | -0.857 |
| F1 — `question` | 0.933 | 0.780 | -0.153 |
| F1 — `shared_info` | 0.842 | 0.522 | -0.320 |

Baseline: `facebook/bart-large-mnli`, zero-shot, given only the definitions in
`label_definitions.txt` (`baseline_results.json`). Trained: `distilbert-base-uncased`,
fully fine-tuned, seed 42 (`results.json`). Both are scored on the same 56
held-out posts in `test_split.csv`. Difference is Trained minus Baseline.

**What I predicted before I looked:**

I predict that my fine-tuned DistilBERT will outperform the zero-shot BART baseline on overall accuracy, because it was trained on posts from the same fitness communities. However, I expect the improvement to be uneven across labels, particularly personal_update, which had poor performance during Unit 5.

**What the gap actually means:**
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->

My prediction was wrong. The zero-shot baseline beat the fine-tuned model on
every measure. BART reached 89.3% accuracy and DistilBERT 67.9%, a gap of 21.4
percentage points. Macro F1 was 0.878 against 0.434. The largest gap is
`personal_update`. The fine-tuned model found none of the 7 `personal_update`
posts in the test split (F1 0.000), and BART found 6 of them (F1 0.857). As
trained in Unit 5, fine-tuning made this classifier worse than using no
training data at all.

Why this happened is not yet known. Three plausible causes, none of them
tested:

- **Class imbalance.** Only 44 of 371 posts (11.9%) are `personal_update`,
  so the model can reduce its loss by mostly predicting `question` (58.2%).
  The confusion matrix shows it did exactly that.
- **Too few examples.** 259 training posts, 31 of them `personal_update`,
  and only 51 optimizer steps. Validation loss was still falling at epoch 3.
- **Label quality.** 351 of the 371 labels were produced by an AI and never
  reviewed by a person. Two of the three cold-20 disagreements sit on the
  `question`/`personal_update` boundary, the same boundary the model fails on.

**What this comparison does not show.** It does not prove that fine-tuning
doesn't work for this task. The two models differ in more than training.
BART-large-MNLI is a much larger model with different pretraining, including
training on natural-language inference, and it read my label definitions
directly. DistilBERT is a small distilled encoder that learned only from 259
examples. The comparison does show that this fine-tuning run, with this data,
did not beat a strong zero-shot model on these 56 posts. It is also one seed
and one test split, where a single post is worth about 1.8 accuracy points.

---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Test accuracy | ≥ 0.75 | 0.679 | 0.750 | 0.786 | **MISSED** |
| 2. Per-label F1 (`question` / `shared_info` / `personal_update`) | ≥ 0.80 / ≥ 0.60 / ≥ 0.50 | 0.780 / 0.522 / 0.000 | 0.821 / 0.741 / 0.000 | 0.833 / 0.848 / 0.000 | **MISSED** |
| 3. Balance in `labels.csv` (largest / smallest) | ≤ 65% / ≥ 15% | 58.2% / 11.9% | 58.2% / 11.9% | 58.2% / 11.9% | **MISSED** |
| 4. Cold-20 agreement, human vs blind AI | ≥ 15/20 | 17/20 | 17/20 | 17/20 | **MET** |
| 5. Confidence gap (most- minus least-confident third) | ≥ 0.15 | 0.944 − 0.444 = 0.500 | 0.944 − 0.722 = 0.222 | 0.833 − 0.722 = 0.111 | **MISSED** |

From `results_three_seeds_before.json`, written by notebook section 6
(`SEEDS = [42, 7, 2024]`, `RUN_LABEL = "before"`). Each seed gets its own
stratified 70/15/15 split and a fresh `distilbert-base-uncased`, with the
Unit 5 settings unchanged (3 epochs, lr 2e-5, batch 16, max length 128) on
Apple Silicon (MPS). Every test split has 56 posts: 32 `question`, 17
`shared_info` and 7 `personal_update`. Confidence thirds are 18 posts each.
Sections 6–7 were run headless with `run_three_seeds.py`, which executes the
notebook's own cells and leaves `results.json` and `test_split.csv` untouched.

- **Criteria 3 and 4 are properties of the dataset, not of a training run.**
  They were not remeasured per seed. The same value is repeated in each
  column (balance from `labels.csv`, agreement from
  `data/cold20_agreement.json`).
- **Criterion 1:** seed 7 lands exactly on 0.750 and meets the target. Seed
  42 does not, so the criterion is missed.
- **Criterion 2:** `personal_update` F1 is 0.000 on every seed. `question` and
  `shared_info` meet their targets on seeds 7 and 2024 but not on seed 42.
- **Criterion 5:** met on seeds 42 and 7 but not on 2024 (0.111). It was
  MET for the single seed in Unit 5 and does not hold across three.
- **Seed 42 reproduced Unit 5.** Its predictions, accuracy, F1 scores,
  confusion matrix and confidence thirds match `results.json` exactly. Only
  the mean confidence differs, in the fifth decimal place
  (0.69757 vs 0.69756), from floating-point nondeterminism on MPS.

The notebook's spread table:

| Measure | Seed 42 | Seed 7 | Seed 2024 | Spread |
|---|---|---|---|---|
| Overall accuracy | 0.679 | 0.750 | 0.786 | 0.107 |
| Macro F1 | 0.434 | 0.520 | 0.561 | 0.127 |
| F1 — `question` | 0.780 | 0.821 | 0.833 | 0.053 |
| F1 — `personal_update` | 0.000 | 0.000 | 0.000 | 0.000 |
| F1 — `shared_info` | 0.522 | 0.741 | 0.848 | 0.327 |

**Stability:** accuracy moved 0.107 across seeds, more than 10 points. By the
notebook's own rule, this test set is too small or too lopsided for a stable
measure. The spread comes almost entirely from `shared_info` (F1 from 0.522
to 0.848). The `personal_update` failure is the one stable result: F1 0.000
on all three seeds.

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

Seed 42, as printed by notebook section 7 (rows are true labels, columns are
predictions):

| true \ predicted | question | personal_update | shared_info |
|---|---|---|---|
| **question** | 32 | 0 | 0 |
| **personal_update** | 7 | 0 | 0 |
| **shared_info** | 11 | 0 | 6 |

The other two seeds, from the same file:

| Seed | `question` → `shared_info` | `personal_update` → `question` | `shared_info` → `question` | All other off-diagonal cells |
|---|---|---|---|---|
| 7 | 0 | 7 | 7 | 0 |
| 2024 | 2 | 7 | 3 | 0 |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->

On seed 42, 11 real `shared_info` posts were called `question`, and none went
the other way. Almost every error, on every seed, runs in one direction:
towards `question`, the majority label (58% of the data).

- **`shared_info` → `question`:** this is where the seeds differ most (11,
  7, 3). The model learns the `question`/`shared_info` boundary on some
  splits and not on others. Only 2 posts in all three seeds went from
  `question` to `shared_info`.
- **`personal_update` → `question`:** the bigger problem. All 7
  `personal_update` posts are called `question` on every seed, 21 of 21,
  and the model never predicts `personal_update` at all. It has not learned
  the `question`/`personal_update` boundary in either direction. It folds
  the smallest label into the largest one.



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 | Test accuracy | ≥ 0.75 | **MISSED** | Seed-dependent: 0.679 / 0.750 / 0.786. Seed 42 is below target. |
| 2 | Per-label F1 | `question` ≥ 0.80, `shared_info` ≥ 0.60, `personal_update` ≥ 0.50 | **MISSED** | Seed-dependent: `personal_update` is 0.000 on all three seeds. `question` (0.780) and `shared_info` (0.522) also miss on seed 42. |
| 3 | Balance in `labels.csv` | largest ≤ 65%, smallest ≥ 15% | **MISSED** | Fixed property of the dataset, the same for every seed: `personal_update` is 11.9%. |
| 4 | Cold-20 agreement | ≥ 15/20 | **MET** | Fixed property of the dataset: 17/20 (`data/cold20_agreement.json`). |
| 5 | Confidence gap | ≥ 0.15 | **MISSED** | Seed-dependent: 0.500 / 0.222 / 0.111. Seed 2024 is below target. |

All targets are the original ones in `criteria.md`, unchanged. Seed-dependent
numbers come from `results_three_seeds_before.json`.

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->

> ⚠️ **The staff agreement check is pending, not measured.** My labels for
> the 30 staff posts are committed (`my_staff_labels.csv`), but the staff
> answers (`data/staff_labels.csv`) have not arrived. The only consistency
> evidence below is the cold-20 comparison (my hand labels vs the AI's blind
> labels). It covers 20 posts, and only 3 of them are `personal_update`.

**Evidence about the labels, as far as it goes.** Each seed's test split has
56 posts, and only 2 to 5 of them were labelled by hand. The rest are
unreviewed AI labels. So every score here measures the model against the
AI's reading of my taxonomy. The cold 20 are the only place where that reading
was checked against mine. They agree on 17 of 20. The three disagreements run
in one direction:

- On 2 of the 3 posts I called `personal_update`, the AI said `question`.
- On 1 post I called `question`, the AI said `shared_info`.
- The AI never moved a post into `personal_update`.

That is the same direction as the model's main error. It suggests the AI
labels may undercount `personal_update` on the `question` boundary, which
would leave the model even fewer and blurrier examples. But it rests on 2
posts, so it is a lead, not a finding. The staff agreement result is what
would show whether I apply a written rule consistently. It isn't available
yet.

**Criterion 2, `personal_update` F1 0.000 on every seed: mainly too few
examples and a majority-class collapse. A labelling contribution is possible
but not shown.**

- The model never predicts `personal_update` on any seed. All 21 test posts
  across the three seeds went to `question`. A model that never outputs a
  label hasn't learned that label from 31 training examples (12% of
  training) with an unweighted loss, where calling everything `question`
  costs little.
- The task itself is reachable. On the seed-42 test split, zero-shot BART
  got 6 of the same 7 `personal_update` posts right from the definitions
  alone (F1 0.857, Milestone 1). Those 7 gold labels are AI labels. A
  different model reading my definitions picks the same posts out, so the
  boundary is learnable from the text.
- The training is too short to escape the majority label. On seed 7,
  validation accuracy stayed at 0.589 for epochs 1 and 2, which is exactly
  the share of `question` in that validation split (33/56). So the model was
  predicting `question` for every post. Validation loss was still falling at
  epoch 3 on every seed.
- **Why I don't call this mainly a labelling problem:** if the
  `personal_update` labels were simply inconsistent, I'd expect a weak,
  noisy F1, not exactly zero on every seed with not a single prediction.
  Zero predictions is what imbalance plus short training looks like.

**Criteria 1 and 2 on seed 42 (`question` 0.780, `shared_info` 0.522): a
leaning towards `question` that depends on the split.** Always guessing
`question` scores 32/56 = 0.571 on every split. The model beats that only as
far as it learns `shared_info`, and that varies by seed:

| Seed | `shared_info` posts called `question` | `shared_info` F1 |
|---|---|---|
| 42 | 11 of 17 | 0.522 |
| 7 | 7 of 17 | 0.741 |
| 2024 | 3 of 17 | 0.848 |

Accuracy moves 10.7 points across seeds, with the same settings and the same
test-set size and label mix. So seed 42's misses are partly a hard split and
partly an under-trained model. Neither number is stable enough to call the
model either good or bad at this boundary. Because `question` is always
over-predicted, its precision suffers (0.64 on seed 42), and that is why its
F1 misses on seed 42 despite perfect recall.

**Criterion 3, balance 11.9% against a 15% floor: too few examples
collected.** This is a dataset fact and training can't change it. The
target assumed about 65 `personal_update` posts, based on the 40-post sample;
the communities produced 44. The unreviewed AI labels could also be
undercounting the label (see above). This criterion can only be fixed by
collecting more data or by a human review of the 351 AI labels.

**Criterion 5, confidence gap 0.111 on seed 2024: small-sample noise, plus
a model that is confidently wrong on one label.** With 18 posts per third,
each post is worth 0.056, so 0.111 is 2 posts. The gap shrinks as accuracy
rises (0.500, 0.222, 0.111): the least-confident third is right 0.722 of
the time on both seeds 7 and 2024, so there are few errors left to separate.
The results files don't record per-post confidence, so I can't tell which
third the `personal_update` errors fall in. I don't read this as evidence
that confidence is meaningless. I read it as evidence that a 0.15 gap can't
be measured reliably on 56 posts.


---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** ___ / 30 = ___%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

**1.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



---

## The Improvement

**What I changed:** one thing, the training loss. Cross-entropy is now
weighted by inverse label frequency in each seed's **training split**, using
`n / (k × count)`. That gives `question` ≈ 0.57, `shared_info` ≈ 1.12 and
`personal_update` ≈ 2.79, so a missed `personal_update` costs about five
times as much as a missed `question`.

- It's done in a new notebook cell, *6a. The improvement*, as a `Trainer`
  subclass (`WeightedLossTrainer`). The Unit 5 cells are unchanged, and
  setting `CLASS_WEIGHTED = False` reproduces the before-run.
- Everything else is the same: data, labels, splits, seeds, model, epochs,
  learning rate, batch size and max length.
- The weights come from training counts only. The test split was not used to
  choose or tune anything.

**Which diagnosis pointed at it:** criterion 2's `personal_update` failure. The
model never predicted the label on any seed, and on seed 7 it predicted only
`question` for two epochs. That pattern points to the minority label being
drowned out by an unweighted loss. BART finding 6 of 7 of those test posts
shows the label is learnable from the text. Relabelling was the other
candidate, but the evidence for a labelling problem is 2 posts on the cold 20,
and the staff agreement result isn't available. So I chose the change the
evidence supports best.

**What I expect (written before the after-run):** `personal_update` recall
should rise above zero on at least some seeds. Some `question` posts will
probably start being called `personal_update`, so `question` F1 may drop a
little. I don't expect overall accuracy to change much. Criterion 3 (balance)
and criterion 4 (agreement) can't move, because training doesn't touch the
dataset.

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**
