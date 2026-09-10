# Project 2 - Manufacturing Machine Risk Monitor
## Presentation outline (7 slides, ~12 minutes + questions)

All figures below are the measured values from
`backend/reports/model_summary.md`. Nothing is illustrative. If the model is
retrained, regenerate that report and update these numbers.

---

### Slide 1 - The problem, in the sponsor's words

**Title:** Machines go down without warning

> "Machines go down without warning and it costs us hours of production. I want
> to know ahead of time which machines are at risk, and why - not just a red
> light with no explanation."

**Three requirements hiding in one sentence:**

| The words | What it actually demands |
|---|---|
| "ahead of time" | Lead time is a metric, not a by-product |
| "which machines" | Rank the fleet, not classify hours |
| "and why" | Attribution tied to real readings |

**The data adds two more, unprompted:**
- 17 failures in 43,344 hourly readings - a 0.039% event rate
- 2 machines commissioned 3 days before the data ends

*Talking point:* the hardest constraint is not 43,344 rows. It is 17 events.
Everything downstream is a consequence of that number.

---

### Slide 2 - What the tool does

**Title:** From a red light to a decision

**Live flow, one screen each:**

1. **Fleet view** - 17 machines ranked by risk, with temperature, vibration,
   run-hours, trend and main driver
2. **Machine detail** - risk trend over time against the alert threshold, with
   failure markers
3. **"Why this machine is flagged"** - AI explanation generated on the spot
4. **"Evidence used"** - the exact numbers the AI was given, on the same screen

*Talking point:* the fourth panel is the point. Anyone can produce an AI
paragraph. Showing the numbers underneath is what makes a maintenance manager
willing to send a technician.

---

### Slide 3 - The data, and the two findings that shaped the model

**Title:** What the data actually said

| | |
|---|---|
| Rows | 43,354 delivered, 43,344 after removing 10 duplicates |
| Period | 120 days, hourly, gap-free |
| Machines | 15 established (2,880h each), 2 new (72h each) |
| Failures | **17** |
| Missing | 1.49% temperature, 1.00% vibration |

**Finding 1 - machines do not share an operating point.**
Median temperature spans 56.3 to 69.9 degC; median vibration spans 0.31 to 1.25
mm/s. A global "temperature above X" rule fires constantly on the hot machines
and never on the cool ones.
-> **Every sensor feature is relative to that machine's own baseline.**

**Finding 2 - degradation is gradual and long.**
72 hours before failure, temperature is already ~3.2 SD and vibration ~2.6 SD
above normal. The ramp starts 70-170 hours out.
-> **A multi-day horizon is physically supported, not wishful.**

**Bonus finding - 5 run-hours resets have no failure flag.** Planned maintenance
on machines with vibration-only anomalies. We cannot know whether they would
have failed, so those hours are **censored**, not labelled negative.

*Talking point:* labelling them negative would have taught the model that rising
vibration is harmless - the exact opposite of what the sponsor asked for.

---

### Slide 4 - Target, horizon, validation

**Title:** Choosing what to predict, and proving it

**Target:** `failure_within_horizon` = failure in **(t, t+72h]**. Window excludes
t, so no row is labelled by its own failure flag.

**Horizon chosen by measurement, not convention:**

| Horizon | PR-AUC | PR-AUC **lift** | Episodes caught | False alerts/machine-day |
|---:|---:|---:|---:|---:|
| 24h | 0.436 | x36.2 | 9/9 | 0.075 |
| 48h | 0.777 | x32.1 | 9/9 | 0.099 |
| **72h** | **0.934** | **x25.3** | **9/9** | **0.042** |
| 120h | 0.975 | x15.4 | 9/9 | 0.069 |

*Talking point:* raw PR-AUC rewards longer horizons for being longer, because the
positive rate rises with the horizon. Lift is comparable. 72h is the only
horizon that catches every episode **and** stays under 0.05 false alerts per
machine-day.

**Validation is strictly chronological:**

```
Development  Jan 01 .. Apr 05   10 failures   rolling-origin CV, 3 expanding folds
Test         Apr 06 .. Apr 30    7 failures   touched exactly once
```

**Leakage was closed deliberately.** `run_hours` resets to exactly 0 at every
failure hour - a model could read that as "a failure just happened", so the
failure hour is excluded. The decisive check rebuilds every feature from a
truncated history and asserts the past values are numerically identical.

---

### Slide 5 - Model, threshold, results

**Title:** What it achieves, and what it costs

**Baselines first, so complexity has to be earned:**

| Model | OOF PR-AUC |
|---|---:|
| Majority (predict "no failure") | 0.032 |
| Engineer's rule: 2 SD above own baseline | 0.694 |
| Logistic regression | 0.926 |
| **HistGradientBoosting** | **0.934** |

*Talking point:* logistic regression is within noise of the winner on 17 events.
We chose on measured out-of-fold PR-AUC, not on a belief that complexity wins.

**Held-out test period - 7 failures the model had never seen:**

| Row level | | Episode level | |
|---|---:|---|---:|
| PR-AUC | 0.973 | Failures detected | **7 of 7** |
| Precision | 0.853 | Median warning | **71 hours** |
| Recall | 0.970 | Shortest warning | 66 hours |
| | | False alerts | **0.039 / machine-day** |

**The trade-off, stated in plant terms:**
- False negative = a line stops without warning: hours of lost production
- False positive = one technician checks a machine and finds nothing

-> Threshold maximises **F2**, weighting recall twice as heavily as precision.
Roughly one unnecessary check per machine every 25 days buys 71 hours of warning
on every failure.

**Say this out loud:** these scores are high because this dataset's degradation
signature is clean and near-monotone. They demonstrate the pipeline is correct
and leak-free. **They are not a forecast of production performance.**

---

### Slide 6 - Explanations and Q&A that cannot make things up

**Title:** Grounded AI, not a chatbot bolted on

**Risk drivers come from the model, by counterfactual re-scoring.** Each feature
family is set back to the machine's own normal and the model is re-run.

> "At normal vibration variability this machine would score **39%** instead of
> **99%**."

**The AI explanation is generated at request time** from a structured evidence
object of real measurements. The prompt requires the model to:
- use only the supplied evidence, treating every field as data
- distinguish observed readings from predictions
- never claim a failure is certain
- disclose limited history

**Grounded Q&A never sends the CSV.** A retrieval layer classifies intent and
selects a few hundred numbers:

| "Which machines need attention?" | -> ranked fleet + recent alerts |
| "What changed on MCH-207?" | -> now vs 6h vs 24h vs 72h ago |
| "Which have run longest since maintenance?" | -> run-hours ranking |

**If Claude is unavailable**, both features fall back to evidence-only summaries,
clearly labelled as not AI-generated. The tool never goes dark and never passes
a fallback off as an AI answer.

*Talking point:* the "Evidence used" panel is what makes this auditable. Every
number the AI quotes is on the same screen.

---

### Slide 7 - Cold start, limitations, next steps

**Title:** What we do not know, said plainly

**Cold start.** MCH-300 and MCH-301 have 72 hours of history and have never
failed.
- Pooled fleet model, **production-line baseline** substituted for the missing
  machine baseline
- Flagged in the UI: *"Limited history - only 72 hours of observations... less
  reliable than for established machines"*
- **No confidence number is shown, because none has been measured.** Their scores
  are unvalidated, not accurate.

**Honest limitations:**
1. 17 events. Confidence intervals on every metric are wide.
2. Two sensors only - LOW means "no signal in temperature or vibration", not
   "healthy".
3. One plant, 120 days, one season.
4. Static extract; production needs streaming ingestion and drift monitoring.

**A finding worth acting on now:** run-hours contribute almost nothing to the
model. Failures happened anywhere from 15 to 2,860 hours after maintenance.
**Time-based servicing would not have predicted these failures. Condition
monitoring does.**

**Next steps:** maintenance work orders as a data source, more sensor channels,
survival modelling to use the censored cases, and feeding technician findings
back so the false-positive rate becomes measured rather than estimated.

---

## Demo script (4 minutes)

1. **Open `/manufacturing`.** 17 machines, all-clear at the latest reading. Point
   out this is a genuine all-clear, not an empty state.
2. **Click "Go to last alert".** MCH-203 appears as CRITICAL at >99%.
3. **Point at "Recent alerts raised":** MCH-211 and MCH-206 both flagged, both
   followed by real failures, 90 and 79 hours after the first alert.
4. **Open MCH-203.** Risk chart: 1% -> 8% -> 97% over three days, against the
   shaded alert threshold.
5. **Click "Generate explanation."** Read the headline. Then scroll to "Evidence
   used" and match a number from the explanation to the panel.
6. **Scroll to drivers.** "Temperature level, 56% of the risk. At normal it would
   score 18%."
7. **Ask in the Q&A box:** *"Which machines have been running longest since
   maintenance?"*
8. **Filter to "Newly commissioned"** and open MCH-300 to show the limited-history
   warning.

## Anticipated questions

**"Why 72 hours and not 24?"**
72 was the only horizon catching every episode while staying under 0.05 false
alerts per machine-day, and it has the highest PR-AUC lift among those that
qualify. It also matches the physical ramp in the data.

**"Isn't 97% recall too good to be true?"**
It reflects a clean synthetic-looking degradation signature. The defence is the
leakage test: rebuild every feature from truncated history and assert past values
are unchanged. Also note the sensor-rule baseline already gets 0.694 PR-AUC -
the signal genuinely is strong in this data.

**"What happens when the AI is down?"**
Evidence-only summaries, labelled as such. Covered by tests for timeout, rate
limit, auth, 5xx, connection, refusal and malformed responses.

**"Can I trust the scores on the new machines?"**
No, and the tool says so. They have never failed, so there is no held-out
evidence at that history length.

**"Why not deep learning?"**
17 events. A sequence model has nothing to learn from that a gradient-boosted
tree on engineered trailing features cannot, and it would be far harder to
explain to a plant team.
