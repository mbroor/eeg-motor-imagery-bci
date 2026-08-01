# EEG Motor Imagery Classifier

Classifies imagined left- vs. right-hand movement from EEG using the
[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
(Schalk et al., 2004) — 64-channel EEG from 109 subjects performing real and
imagined limb movements, recorded with the BCI2000 system.

## Pipeline
- Load raw EEG (runs 4, 8, 12 — imagined left/right fist) via **MNE-Python**
- Band-pass filter to the mu/beta bands (8–30 Hz)
- Epoch the signal around movement-imagery cues (1–4s post-cue)
- Extract features and classify with **scikit-learn**

## Results

| Approach | Accuracy |
|---|---|
| Naive log-variance, single subject | 0.38 mean CV (chance) |
| Naive log-variance, pooled (10 subjects) | 0.51 mean CV (chance) |
| CSP (Common Spatial Patterns), single subject | **0.69 mean CV** |
| CSP, one filter pooled across 10 subjects | 0.49 mean CV (chance) |
| CSP, fit individually per subject (10 subjects) | **0.55 average (+/- 0.19), range 0.33–0.96** |

## Why the results look like this (not a bug — this is the real problem)

Raw per-channel log-variance sits at chance: motor imagery signal (mu/beta
ERD/ERS) is spatially diffuse, so no single electrode captures it cleanly.
**CSP** — a supervised spatial filter that reweights electrodes to maximize
the variance difference between classes — recovers real signal for a single
subject (0.69).

But CSP filters are subject-specific: fitting *one* filter across pooled data
from 10 different people collapses back to chance (0.49), since each
person's optimal spatial pattern differs (skull thickness, cortical folding,
electrode contact). Fitting CSP **individually per subject** instead gives an
average of 0.55, with a wide spread (0.33–0.96) — consistent with
**"BCI illiteracy"**, a documented phenomenon where a meaningful fraction of
people show weak motor-imagery signal even with a well-tuned pipeline.

## Limitations
- Small sample per subject (45 epochs), so individual scores carry real
  variance — not a precise measurement of any one person's "true" decodability
- Only 10 of 109 available subjects used so far

## Next steps
- Scale to more subjects; check whether per-subject accuracy correlates with
  anything measurable (electrode impedance, session length)
- Hyperparameter tuning (`GridSearchCV`) on CSP component count and k in k-NN
- Compare k-NN vs. logistic regression vs. SVM as the final classifier
- Try Filter-Bank CSP (multiple frequency bands) instead of one fixed 8–30 Hz band

## How to run
Open `01_motor_imagery_classifier.ipynb` in Google Colab — all dependencies
install in the first cell.
