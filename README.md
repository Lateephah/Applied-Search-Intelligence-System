# Which Content Pages Should You Refresh First?

A validated content-refresh priority model built on 30,000 real pages from the FlyRank ML
Internship dataset — compared honestly against a transparent rule baseline, under a validation
split designed to catch the exact way "it worked in testing" projects usually lie.

**Read the paper:** https://lateephah.github.io/Applied-Search-Intelligence-System/
*(if this link 404s, check `submission/paper_url.txt` in this repo for the current one)*

---

## What this is, and who it's for

Content teams with more pages than editorial time usually decide what to refresh first using a
rule someone made up and nobody ever checked — "it's old, it still gets traffic, fix it." This
project is that rule, built properly and tested against real outcomes, then challenged with a
validated model to see if it earns the right to replace it.

**Built for:** anyone doing content-ops prioritization at scale, or anyone evaluating this as a
sample of applied, honestly-validated ML work — a hiring manager, a reviewer, a future me.

---

## Setup (from zero, no assumptions)

```bash
git clone https://github.com/Lateephah/Applied-Search-Intelligence-System.git
cd Applied-Search-Intelligence-System
pip install -r requirements.txt
```

No API keys, no gated dataset access, no accounts. Everything runs on the anonymized starter CSV
already committed at `data/raw/content_refresh_anonymized.csv` (~30,000 rows, no client names or
raw URLs).

**Reproduce the whole pipeline end to end:**

```bash
jupyter notebook work/notebooks/capstone.ipynb
# Run all cells (Kernel -> Restart & Run All)
```

This single notebook rebuilds the population, the baseline rule, the validated model, the
random-vs-grouped split comparison, the leakage check, and the action-playbook summary — every
number in the paper traces back to this notebook or one of the weekly notebooks in
`work/notebooks/` it summarizes.

**Or open any individual step directly** (each is a standalone, runnable notebook):

| Notebook | What it does |
|---|---|
| `work/notebooks/w04_baseline_score.ipynb` | The transparent rule baseline + the 3 signal checks it's built from |
| `work/notebooks/w05_model.ipynb` | Logistic Regression / Random Forest, trained and compared to the rule |
| `work/notebooks/w06_validation_audit.ipynb` | The random-vs-grouped split audit and the leakage-injection test |
| `work/notebooks/w07_action_playbook.ipynb` | The deployed, reason-coded action queue + content archetypes |
| `work/notebooks/capstone.ipynb` | Condensed end-to-end rerun; this is what the paper mirrors |

---

## Usage example

The pipeline's real output is a ranked CSV an editor can open directly. After running
`w07_action_playbook.ipynb`, `work/outputs/action_playbook_queue.csv` contains one row per page:

```python
import pandas as pd

queue = pd.read_csv("work/outputs/action_playbook_queue.csv")

# The pages worth an editor's first hour -- sorted by economic value at risk,
# not raw probability (see "Limitations" for why that distinction matters).
top_priority = queue.sort_values("value_weighted_score", ascending=False).head(20)
print(top_priority[["content_id", "reason_code", "action", "decline_probability"]])
```

Every row also carries a `reason_code` (e.g. `stale_content`, `weak_click_through`) an editor can
read without opening a notebook, and an `action` tier (`PRIORITY_REVIEW`, `SCHEDULE_REVIEW`,
`MONITOR`) for a fast triage pass.

---

## Architecture

```text
data/raw/content_refresh_anonymized.csv   (30,000 pages, 32 clients, trailing-90-day metrics)
              |
              v
   [ population filter ]   impressions_90d > 0  AND  content_age_days >= 90
              |
              v
   [ baseline rule ]        w04: stale + visible + CTR-gap  ->  transparent score
              |
              v
   [ signal audit ]         w04/w06: bucket checks, denominator traps, leakage-injection test
              |
              v
   [ model ]                w05: Logistic Regression vs Random Forest
              |
              v
   [ honest validation ]    w06: GroupShuffleSplit by client_id (0 client overlap),
              |             random-split-vs-grouped-split comparison
              v
   [ action playbook ]      w07: reason codes, archetypes, value-weighted re-rank,
              |             human-review + no-go rules
              v
   [ capstone + paper ]     capstone.ipynb condenses all of the above;
                            docs/index.html is the deployed write-up
```

---

## Eval results (validated, not the tuning-population number)

The number that matters is the one measured on data the model never got to memorize: a
`GroupShuffleSplit` held out by `client_id`, 0 of 32 clients appearing in both train and test.

| Model | precision@20 | precision@50 | precision@100 | ROC-AUC |
|---|---|---|---|---|
| Rule baseline (w04) | 0.500 | 0.520 | 0.470 | 0.489 |
| **Logistic Regression** | **0.650** | **0.640** | **0.670** | 0.577 |
| Random Forest | 0.600 | 0.580 | 0.530 | 0.602 |
| *(test-slice base rate)* | *0.511* | *0.511* | *0.511* | — |

The gap that matters more than any single number: the same rule scored **0.650** at
precision@20 back when it was measured on its own 30,000-row tuning population — the number
above (0.500) is what it actually does on clients it's never seen. A naive random split (letting
clients leak across train/test) inflates the *validated model's* own ROC-AUC from 0.577 to 0.704
and precision@20 from 0.650 to 0.900 — full comparison in `w06_validation_audit.ipynb`.

---

## Limitations (stated, not buried)

- **Proxy label, not ground truth.** The decline label is a threshold on a 30-day-vs-prior-30-day
  swing — noisy at low impression counts (see `w05`'s false-negative example: a label built on a
  single impression).
- **One held-out split.** These numbers come from a single client-grouped 80/20 partition, not a
  repeated cross-validation. Real evidence, not a guarantee that holds for every possible split.
- **Cross-sectional, one snapshot.** No claim here is "refreshing this page will increase
  traffic" — the honest form is decision-support: this page looks worth reviewing first.
- **Not yet validated on a brand-new client.** The random-vs-grouped gap is itself evidence that
  confidence should be lower for a client outside this training population until spot-checked.
- **Does not model a search engine's ranking algorithm** — it models this portfolio's own
  observed patterns, nothing more.
- **No claims about specific AI writing tools.** `model_used` and `content_type` showed up as
  model features, but per-category sample sizes are small and uneven — not sufficient evidence
  about any one tool's output quality.

---

## Built with AI — what and how

This project was built with **Claude (Anthropic)** as a coding and validation assistant across
every notebook. Concretely: Claude drafted the baseline rule, model pipeline, and playbook code
from my framing decisions; ran and re-ran each notebook to completion so every number in this
README and the paper is a real, executed output, not a guess; and caught (and fixed, on the
record) real bugs along the way — including a data trap in the CTR-vs-position tier medians
(`avg_position == 0` silently sorting into the top tier) and a broken majority-class baseline
computation in the model comparison. I chose the lane, the validation design, the feature
exclusions, and reviewed every claim against the notebook output before it went in the paper.
Where a finding surprised me (the rule failing to generalize to unseen clients; `days_since_last_update`
not making the model's top-10 features), I asked for the honest number rather than the flattering
one, and both are in the paper.

---

## Repo map

```text
data/raw/                 the anonymized starter dataset
work/notebooks/           every weekly notebook + capstone.ipynb
work/outputs/             metrics JSONs (committed) + the regenerated queue CSV (gitignored)
work/figures/             committed charts the paper embeds
docs/                     the deployed paper (index.html + img/)
submission/paper_url.txt  the one-line pointer to the live paper
skills/                   the instruction library used to direct the AI assistant
```

*Data credit: built on the [FlyRank ML Internship dataset](https://flyrank.ai). Code under MIT
(see `LICENSE`); data under `DATA_USE.md`.*
