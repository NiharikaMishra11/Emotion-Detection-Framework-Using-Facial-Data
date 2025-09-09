# Emotion Detection Framework Using Facial Video Data Enhanced with Generative AI

This project detects human emotions from facial video clips using lightweight, CPU-friendly techniques. It relies on classical vision features (LBP/HOG) and simple classifiers (Logistic Regression, SVM, Random Forest). Optional synthetic augmentation (flips, small rotations, brightness/contrast, noise) improves robustness without heavy models.

## Features

- Dataset handling: frame sampling (~1 fps) from .flv/.mp4/.avi files; intensity filter (HI only except NEU; optional MD).
- Preprocessing: face detection + crop (OpenCV Haar cascade), grayscale, resize (48–64), CLAHE normalization.
- Features: Local Binary Patterns (LBP) or Histogram of Oriented Gradients (HOG).
- Models: Logistic Regression, SVM (RBF), Random Forest.
- Generative AI enhancement: synthetic augmentation via flips, small rotations, brightness/contrast jitter, and Gaussian noise (with before/after examples option).
- Evaluation: accuracy, precision, recall, F1, confusion matrix.
- Inference: single-video CLI prediction with confidence.

## Project Structure

```
Emotion Detection Project/
├── data/
│   ├── raw/
│   └── processed/
├── preprocessing/
│   ├── data_preprocessing.py
│   └── lbp_extractor.py
├── models/
│   └── baseline_models.py
├── genai/
│   └── augmentation.py
├── evaluation/
│   └── metrics.py
├── train.py
├── requirements.txt
└── README.md
```

## What the Project Does

- Extracts frames from facial videos (~1 fps), detects and crops the face, normalizes contrast (CLAHE), and converts to grayscale.
- Computes either LBP or HOG features for each frame.
- Trains a classic classifier (Logistic Regression, SVM, or Random Forest) on these features to predict one of six emotions.
- Optionally augments training data with simple transformations to simulate generative enhancements.
- Evaluates with accuracy, precision, recall, F1-score, and a confusion matrix.

## Dataset Used

- CREMA-D subset with six emotions: Angry (ANG), Disgust (DIS), Fear (FEA), Happy (HAP), Neutral (NEU), Sad (SAD).
- All videos in a single folder. Filename format: `ActorID_SentenceID_Emotion_Intensity.ext`.
- Supported extensions: `.flv`, `.mp4`, `.avi`.
- Intensity policy: keep only High (HI) for non-neutral; keep all intensities for Neutral. Optional flag to include Medium (MD).

## Pipeline (Short)

1) Frame Sampling: ~1 frame/sec from each video.
2) Preprocessing: Haar face detection → crop largest face → grayscale → resize (48–64) → CLAHE.
3) Features: LBP or HOG per frame.
4) Training: Logistic Regression, SVM (RBF), or Random Forest on frame features.
5) Augmentation (optional): flips, small rotations, brightness/contrast, Gaussian noise.
6) Evaluation: accuracy, precision, recall, F1, confusion matrix.
7) Inference: single-video prediction with aggregated frame probabilities or majority vote.

## Requirements

- Python 3.9+
- Install dependencies:
  - `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `Pillow`
  - Or simply: `pip install -r requirements.txt`

## Acknowledgments

- CREMA-D dataset
- scikit-learn, OpenCV, scikit-image

