# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written in
unit 5 **before** anything was trained.

**All five are yours this time.** None are given. You've had two projects of
practice.

An acceptance criterion names a number. *"The model is accurate"* is an
opinion. *"Every label has an F1 of at least 0.60 on the held-out set"* is a
criterion.

Under each, write a sentence or two on **why that number**. A reason that says
something about your data or your taxonomy earns credit — *"I picked 0.60 F1
for `reaction` because it's my smallest label and I only have about 50
examples of it"*. A reason that could be attached to any project does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## Pick numbers you can defend

Not numbers that sound impressive. Three labels means a coin-flip guesser gets
about 33%, so a target of 0.40 is barely a target. Your number should sit
somewhere you'd honestly call useful.

**Cover at least three of these five areas.** They're here as prompts, not as a
form to fill in — a criterion that fits none of them is fine if it names a
number.

| Area | A question it could answer |
|---|---|
| Overall accuracy | How often does it need to be right to be worth using? |
| Per-label performance | Is one label allowed to be much worse than the others? |
| Balance | How lopsided can your label counts get before it's a problem? |
| Consistency | If someone else labelled the same posts, how often should you agree? |
| Confidence | Should a confident prediction be right more often than an unsure one? |

Two things worth knowing before you pick numbers, because both will affect
whether you hit them:

- **Your smallest label will have the jumpiest score.** If a label has 50
  examples, about 8 land in the test split. An F1 computed on 8 examples moves
  a lot between seeds. A target for that label should be looser than one for
  your biggest label, and saying so is a good reason.
- **Unit 6 tests across three seeds, and the target has to hold across all
  three.** A target of 0.65 against results of 0.71, 0.62, 0.68 is a **miss**.
  Pick with that in mind — it is stricter than it first sounds.

---

## 1. Overall accuracy of at least 0.75 on the held-out test split

Measured from `accuracy` in `results.json`.

**Why this target:** `question` is close to 60% of my posts, so a model that
always says `question` scores about 0.58 without learning anything. 0.75 asks
for a clear margin over that majority guess, and that margin is what makes a
sorter of fitness posts worth using. It doesn't ask for perfection on 374 posts
where the question/personal_update boundary is a judgement call.

---

## 2. Per-label F1 on the held-out test split: `question` at least 0.80, `shared_info` at least 0.60, `personal_update` at least 0.50

Measured from `per_label.<label>.f1` in `results.json`. All three must hold.

**Why this target:** each target is set by the size of its label. `question`
has around 210 examples and about 33 in test, so it should be learned well.
`shared_info` posts mostly look different on the surface (headline titles,
quoted article text), so 0.60 is reasonable at roughly 85 examples.
`personal_update` is my smallest label, at about 65 examples and around 10 in
test. It also sits on my hardest boundary, with `question`, so it gets the
loosest target. Macro F1 alone could hide one label collapsing to zero, which
is why this criterion is per label.

---

## 3. Balance: in `labels.csv`, the largest label is at most 65% of rows and the smallest is at least 15%

Measured from the label counts in `labels.csv` (`python labelling.py check`).

**Why this target:** the brief caps a label at 70%, but my taxonomy sample
already had `question` near 58%. Holding it at 65% gives a little headroom, and
a 15% floor keeps the smallest label above roughly 55 examples. Below that its
test F1 would rest on a handful of posts.

---

## 4. Consistency: on the cold 20, an independent second labeller (the AI, labelling blind after I finished) agrees with my labels on at least 15 of 20 (75%)

Measured in `data/cold20_agreement.json`. The second labels are written before
the comparison is run, and my labels are committed before the second labeller
sees the posts.

**Why this target:** my three decision rules are only useful if someone who
didn't write them can apply them and reach the same answers. This is also the
only check on the pre-labelling workflow: if the AI disagrees with me on more
than a quarter of the cold posts, its 354 pre-labels need much heavier
correction than "review and accept". 75% allows for the question/personal_update
boundary being genuinely hard on a few posts.

---

## 5. Confidence: accuracy on the model's most-confident third of the test split is at least 0.10 higher than on its least-confident third

Measured as `confidence.accuracy_most_confident_third` minus
`confidence.accuracy_least_confident_third` in `results.json`.

**Why this target:** the realistic use here is routing. Confident predictions
get sorted automatically, and unsure ones go to a person. That only works if
confidence separates right from wrong. With about 57 test posts, each third is
about 19 posts, so one post is worth about 0.05. A gap of 0.10 is two posts'
worth, the smallest difference I'd treat as real rather than noise.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     UNIT 6 — read this before you change anything above.

     If a criterion turns out to be BROKEN rather than merely unmet, you can
     revise it, and that earns credit. But never delete or edit the original
     line. Add the revision underneath, like this:

         ## 2. Every label performs acceptably

         The model performs well on all labels.

         **Why this target:** ...

         > **Revised in unit 6:** Every label has an F1 of at least 0.60 on
         > the held-out set.
         >
         > **Why revised:** "performs well" gave me nothing to check. I
         > couldn't produce a verdict from it at all.

     That's a revision because the criterion couldn't be MEASURED.

     Lowering a target because you missed it is not a revision, and it costs
     you the point:

         ✗ "Overall accuracy of at least 0.65" → "at least 0.55", because
            0.65 turned out to be optimistic for 200 examples.

     A number you missed stays where it is, gets diagnosed, and gets a fix
     attempted. That's where the points are.
     ───────────────────────────────────────────────────────────────────────── -->
