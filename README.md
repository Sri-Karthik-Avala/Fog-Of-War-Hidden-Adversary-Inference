# Fog Of War: Hidden Adversary Inference

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 5 |
| Last submission | 2026-06-25 |

## Problem statement

### Background

In competitive multi-agent environments, each side must act on an incomplete picture of
 its adversary. An agent sees only the region it has actively scouted; everything else is
 stale or hidden. Effective play depends on *inferring* the adversary's true posture and
 strength from this fragmentary view — reading faint spatial cues to estimate what cannot
 be directly seen.

This challenge frames that problem as supervised inference on a single observation. You are
 given one agent's fog-limited occupancy grid for a single moment of a match, and must
 recover three properties of the adversary's *true* (fully-observed) state — including parts
 of it that lie outside the observing agent's current visibility. Strong solutions cannot rely
 on what is directly visible; they must learn to reason about hidden structure from partial,
 partially-stale evidence.

### Task

Each example is a single 32×32, 3-channel occupancy grid taken from one agent's
 perspective (the "observer"). The three channels encode, per map cell, the most recent
 time a unit was seen there:

- **Channel 0** — the observer's own forces (fully visible).
- **Channel 1** — the adversary's forces *as currently observed* (cells the observer has
    not scouted are blank, even if the adversary is present there).
- **Channel 2** — neutral resource and terrain context.

From this single grid, predict three targets about the adversary's true state:

| Target | Type | Description |
| --- | --- | --- |
| `target_air` | binary (0/1) | Whether the adversary's combat force is predominantly **aerial** (1) rather than ground/static (0). |
| `target_value` | regression (≥ 0) | The total resource-equivalent value of the adversary's **combat** force, **including units currently hidden** from the observer. |
| `target_phase` | ordinal (0–3) | The temporal stage of the engagement, from early (0) to late (3). |

The targets describe the adversary's *true* state, not what is visible in the grid. The
 visible channel collapses unit identities together and omits whatever the observer has not
 scouted, so neither the adversary's force composition nor its full size can be read off
 directly — they must be inferred.

### Evaluation Metric

Submissions are scored with a composite of the three targets:

```
score = 0.30 * F1_air  +  0.50 * value_score  +  0.20 * phase_score
```

- **`F1_air`** — macro-averaged F1 over the binary `target_air` task.
- **`value_score`** — `max(0, 1 - SS_res / SS_tot)`, the coefficient of determination (R²)
    computed on `log(1 + target_value)`, clipped to `[0, 1]`. A mean predictor scores 0.
- **`phase_score`** — `max(0, κ)`, the quadratic-weighted Cohen's kappa on `target_phase`.

Each component is clipped to `[0, 1]`; the final score is in `[0, 1]`, higher is better.
 The heaviest weight sits on the hidden-force-value target because it is the property least
 recoverable from the visible grid.

Score interpretation (preliminary — to be confirmed against the reference baseline):

- ~0.25 — trivial constant/mean predictions
- ~0.40 — weak signal extracted
- ~0.55 — solid partial-observation inference
- ~0.70 — strong solution

### Data

- **~18,000** training grids with labels; **~4,300** test grids without labels.
- Images are 32×32 RGB PNGs (3 channels as described above), one per example.
- Train and test are drawn from **disjoint sources**: the test set comes from environments
    and adversary pairings that never appear in training. A model that memorizes
    surface patterns from training will not transfer; the task rewards genuine inference.
- `target_air` is imbalanced (the aerial class is the minority). `target_phase` is roughly
    balanced across its four buckets.

### File Structure

```
dataset/public/train_images/   ~18,000 PNG grids named <id>.png
dataset/public/test_images/    ~4,300 PNG grids named <id>.png
dataset/public/train.csv       id, target_air, target_value, target_phase
dataset/public/test.csv        id   (no targets)
dataset/public/sample_submission.csv   id + placeholder predictions
```

### Submission Format

A CSV with exactly one row per test `id` and **exactly these four columns** (no extra
 columns):

| Column | Type | Notes |
| --- | --- | --- |
| `id` | string | must match an id in `test.csv` |
| `target_air` | int | 0 or 1 |
| `target_value` | float | ≥ 0 (scored on a log scale) |
| `target_phase` | int | 0, 1, 2, or 3 |

Example:

```
id,target_air,target_value,target_phase
a1b2c3d4e5f6,0,1850.0,1
0f9e8d7c6b5a,1,12400.5,3
4455aabbccdd,0,300.0,0
```

Your submission must contain exactly one row for every test `id` — no more and no fewer, with
 a valid value in every column. Missing ids, duplicate ids, ids that are not in the test set, a
 missing required column, any extra column, or any empty / non-numeric prediction value cause
 the submission to be rejected.

### What Not To Use

Using any of the approaches below is grounds for rejection on review, regardless of score.

- **Image reverse-engineering.** Do not match `train_images/` or `test_images/` files back to
    any upstream source archive via pixel hashing, perceptual hashing, reverse image search, or
    filename reconstruction. The anonymised image IDs must be treated as opaque — recovering the
    original match, environment, faction, or unit identities behind any grid from external
    sources is prohibited.
- **External image archives.** Do not download any external dataset, replay collection, or
    related image archive at training or inference time to reconstruct labels or source
    identities. Only `train_images/` together with `train.csv` may be used for supervised
    training.
- **Hard-coded or rule-based prediction.** Targets must come from a model learned on the
    provided training data — no hard-coded answer tables, no hash-keyed lookup dictionaries, and
    no rule-based pipelines that emit targets without learning from `train_images/`.
- **Grader or filesystem exploitation.** Do not probe for the private answers file, read any
    hidden split or group field, or use any channel other than the provided public files
    (`train_images/`, `train.csv`, `test_images/`, `test.csv`, `sample_submission.csv`).
