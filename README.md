# Emotion Detection Framework Using Facial Video Data Enhanced with Generative AI

A lightweight framework for emotion detection from facial video clips using classical vision features (LBP/HOG) and simple classifiers (Logistic Regression, SVM, Random Forest). It includes optional augmentation (generative AI-style synthetic variation), face-cropping with CLAHE, evaluation utilities, and single-video inference.

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

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train (lightweight)

HOG + SVM, include medium intensity, with augmentation (good balance of speed/accuracy):
```powershell
py train.py --dataset_path "C:\\Users\\mishr\\Downloads\\Crema-D Subset" --feature hog --model svm \
  --include_md --augment --max_frames_per_video 6 --test_size 0.3 --img_size 64 --hog_ppc 4 --hog_cpb 2
```

Faster baseline (LBP + LR):
```powershell
py train.py --dataset_path "C:\\Users\\mishr\\Downloads\\Crema-D Subset" --feature lbp --model lr --max_frames_per_video 3
```

Notes:
- Match inference parameters (`--img_size`, `--hog_ppc`, `--hog_cpb`) to the values used in training.
- Supported video extensions: .flv, .mp4, .avi.

### Single-Video Inference

Use the saved artifacts from training (`outputs/model_*.joblib`, `outputs/scaler_*.joblib`). Example:
```powershell
py train.py --predict_video "C:\\Users\\mishr\\Downloads\\Crema-D\\1091_IEO_SAD_HI.flv" --feature hog --model svm \
  --img_size 64 --hog_ppc 4 --hog_cpb 2 --model_path ".\\outputs\\model_hog_svm.joblib" --scaler_path ".\\outputs\\scaler_hog.joblib"
```

## Parameters

- `--include_md`: include medium intensity clips (increases data and typically accuracy).
- `--max_frames_per_video`: frames sampled per video; 5–8 is a good range.
- `--img_size`: preprocessing square size (48–64 recommended).
- `--hog_ppc`, `--hog_cpb`: HOG cell settings; smaller cells capture more detail at higher feature dimensionality.
- `--svm_c`, `--svm_gamma`: SVM hyperparameters (`gamma` accepts `scale`, `auto`, or a float).

## Outputs

- Saved models/scalers in `outputs/`.
- Logs printed to console and saved in `training.log`.

## Acknowledgments

- CREMA-D dataset
- scikit-learn, OpenCV, scikit-image

