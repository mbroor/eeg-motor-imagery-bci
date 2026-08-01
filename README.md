
# EEG Motor Imagery Classifier

Classifies imagined left- vs. right-hand movement from EEG using the
PhysioNet EEGBCI Motor Movement/Imagery Dataset.

## What this does
- Loads raw EEG (64 channels, PhysioNet EEGBCI dataset) via MNE-Python
- Band-pass filters to the mu/beta bands (8–30 Hz)
- Epochs the data around movement-imagery cues
- Extracts log-variance per channel as a feature vector
- Trains a k-NN classifier (scikit-learn) to distinguish left vs. right
- Evaluates with 5-fold cross-validation
- 
## Results
Aggregated across 10 subjects (subjects 1–10, runs 4/8/12), giving 450 epochs
total, each described by 64 log-variance features (one per EEG channel):
- Single 80/20 train/test split: 0.48 accuracy
- 5-fold cross-validation: 0.51 mean accuracy (+/- 0.02)
  - Fold scores: `[0.49, 0.52, 0.48, 0.53, 0.52]`
This is close to chance level for a binary task (0.50), which is an honest
first result for a simple log-variance + k-NN baseline on raw, unscaled,
single-band features. It establishes a working end-to-end pipeline —
data loading → filtering → epoching → feature extraction → classification
→ cross-validation — that later, stronger models can be benchmarked against.

## Why this matters
Motor imagery decoding — inferring which movement someone is imagining,
purely from their EEG — is the core building block behind non-invasive
brain-computer interfaces (BCIs), e.g. for controlling prosthetics,
wheelchairs, or communication devices in patients with motor impairments.
This project reproduces that basic pipeline on real, publicly available
EEG recordings (not simulated data), using the same mu/beta band
sensorimotor rhythms that real BCI systems exploit.

## Next steps
- Common Spatial Patterns (CSP) for better features
- Hyperparameter tuning (GridSearchCV)
- Compare k-NN vs. logistic regression vs. SVM

