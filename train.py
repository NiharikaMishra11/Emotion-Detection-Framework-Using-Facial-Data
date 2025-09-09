"""
Lightweight Training Script for Emotion Detection Framework

This script implements a CPU-friendly pipeline:
- Preprocess frames from videos (1 fps, 48x48 grayscale)
- Extract LBP or HOG features
- Train classic classifiers (Logistic Regression, SVM, Random Forest)
- Evaluate with accuracy, precision, recall, F1, and confusion matrix
- Optional lightweight augmentation (flips, rotation, brightness, noise)
- Single-video inference via --predict_video

Compatible with CREMA-D subset filenames:
ActorID_SentenceID_Emotion_Intensity.(mp4|avi|flv)
Emotion ∈ {ANG, DIS, FEA, HAP, NEU, SAD}
Intensity ∈ {LO, MD, HI, XX}; keep only HI, except keep all for NEU
"""

import os
import sys
import logging
from pathlib import Path
import argparse
import random
from typing import List, Tuple

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

EMOTION_MAP = {"ANG": 0, "DIS": 1, "FEA": 2, "HAP": 3, "NEU": 4, "SAD": 5}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}
VALID_EXTS = {".mp4", ".avi", ".flv"}

# Haar cascade for face detection (use OpenCV's built-in path)
HAAR_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
FACE_CASCADE = cv2.CascadeClassifier(HAAR_PATH)

# ------------------------
# Augmentation (Lightweight)
# ------------------------

def genai_augment(img: np.ndarray) -> np.ndarray:
    """Apply lightweight synthetic augmentations to simulate data diversity."""
    h, w = img.shape[:2]
    out = img.copy()
    # random flip
    if random.random() < 0.5:
        out = cv2.flip(out, 1)
    if random.random() < 0.2:
        out = cv2.flip(out, 0)
    # small rotation
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # brightness/contrast
    if random.random() < 0.7:
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.uniform(-15, 15)    # brightness
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    # gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 8, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

# ------------------------
# Preprocessing
# ------------------------

def parse_labels_from_filename(filename: str) -> Tuple[str, str]:
    """Extract (emotion, intensity) from filename ActorID_SentenceID_Emotion_Intensity.ext"""
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    if len(parts) < 4:
        return None, None
    emotion = parts[-2].upper()
    intensity = parts[-1].upper()
    return emotion, intensity


def should_keep_clip(emotion: str, intensity: str, include_md: bool) -> bool:
    if emotion is None or intensity is None:
        return False
    if emotion == 'NEU':
        return True
    if include_md:
        return intensity in ('HI', 'MD')
    return intensity == 'HI'


def preprocess_frame(frame_bgr: np.ndarray, target_size=(48, 48)) -> np.ndarray:
    """Detect face, crop largest, convert to grayscale, resize to target, apply CLAHE."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(24, 24))
    if len(faces) > 0:
        # choose the largest face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face = gray[y:y+h, x:x+w]
    else:
        face = gray
    resized = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    return enhanced


def frame_generator(video_path: str, target_size=(48, 48), fps_sample=1):
    """Yield processed frames (face-cropped, CLAHE, grayscale) at ~1 fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 25
    step = max(1, int(fps / fps_sample))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            processed = preprocess_frame(frame, target_size)
            yield processed
        idx += 1
    cap.release()

# ------------------------
# Feature extractors
# ------------------------

def extract_lbp_features(img: np.ndarray, radius=1, n_points=8) -> np.ndarray:
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    # Histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_hog_features(img: np.ndarray, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) -> np.ndarray:
    feat = hog(
        img,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    return feat.astype(np.float32)

# ------------------------
# Dataset builder
# ------------------------

def build_dataset(
    dataset_path: str,
    feature_type: str = 'lbp',
    max_frames_per_video: int = 3,
    augment: bool = False,
    save_aug_example_dir: str = None,
    include_md: bool = False,
    img_size: int = 48,
    hog_pixels_per_cell: int = 8,
    hog_cells_per_block: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate videos, extract frames and features, return X, y arrays."""
    X, y = [], []
    paths = []
    dataset_path = str(dataset_path)
    video_files = [str(Path(dataset_path) / f) for f in os.listdir(dataset_path)
                   if os.path.splitext(f)[1].lower() in VALID_EXTS]

    if not video_files:
        raise FileNotFoundError("No video files found. Supported: .mp4, .avi, .flv")

    target_size = (img_size, img_size)
    ppc = (hog_pixels_per_cell, hog_pixels_per_cell)
    cpb = (hog_cells_per_block, hog_cells_per_block)

    random.shuffle(video_files)
    for vpath in video_files:
        emotion, intensity = parse_labels_from_filename(vpath)
        if emotion not in EMOTION_MAP:
            continue
        if not should_keep_clip(emotion, intensity, include_md):
            continue

        label = EMOTION_MAP[emotion]
        frame_count = 0
        for frame in frame_generator(vpath, target_size=target_size, fps_sample=1):
            use_img = frame
            if augment:
                aug = genai_augment(frame)
                use_img = aug
                if save_aug_example_dir and frame_count == 0:
                    os.makedirs(save_aug_example_dir, exist_ok=True)
                    before_path = os.path.join(save_aug_example_dir, f"before_{Path(vpath).stem}.png")
                    after_path = os.path.join(save_aug_example_dir, f"after_{Path(vpath).stem}.png")
                    cv2.imwrite(before_path, frame)
                    cv2.imwrite(after_path, aug)

            if feature_type == 'lbp':
                feat = extract_lbp_features(use_img)
            elif feature_type == 'hog':
                feat = extract_hog_features(use_img, pixels_per_cell=ppc, cells_per_block=cpb)
            else:
                raise ValueError("feature_type must be 'lbp' or 'hog'")

            X.append(feat)
            y.append(label)
            paths.append(vpath)

            frame_count += 1
            if frame_count >= max_frames_per_video:
                break

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    logger.info(f"Built dataset: X shape {X.shape}, y shape {y.shape}")
    return X, y

# ------------------------
# Training and evaluation
# ------------------------

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'lr',
    test_size: float = 0.2,
    random_state: int = 42,
    svm_C: float = 10.0,
    svm_gamma: str | float = 'scale'
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_type == 'lr':
        model = LogisticRegression(max_iter=1000, n_jobs=None)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', C=svm_C, gamma=svm_gamma, class_weight='balanced', probability=True)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    else:
        raise ValueError("model_type must be 'lr', 'svm', or 'rf'")

    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, preds) 
    logger.info(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    logger.info("Confusion Matrix:\n" + str(cm))
    logger.info("Per-class report:\n" + classification_report(y_test, preds, target_names=[IDX_TO_EMOTION[i] for i in range(6)], zero_division=0))

    return {
        'model': model,
        'scaler': scaler,
        'metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }
    }

# ------------------------
# Inference on a single video
# ------------------------

def predict_video(video_path: str, feature_type: str, model_path: str, scaler_path: str,
                  img_size: int = 48, hog_ppc: int = 8, hog_cpb: int = 2):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model/scaler not found. Train first or provide correct paths.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    feats = []
    target_size = (img_size, img_size)
    ppc = (hog_ppc, hog_ppc)
    cpb = (hog_cpb, hog_cpb)

    for frame in frame_generator(video_path, target_size=target_size, fps_sample=1):
        if feature_type == 'lbp':
            feat = extract_lbp_features(frame)
        else:
            feat = extract_hog_features(frame, pixels_per_cell=ppc, cells_per_block=cpb)
        feats.append(feat)

    if not feats:
        raise RuntimeError("No frames could be read from the video for inference.")

    X = np.array(feats, dtype=np.float32)
    Xs = scaler.transform(X)

    # Try probability if available
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(Xs)
        avg_probs = probs.mean(axis=0)
        pred_idx = int(np.argmax(avg_probs))
        confidence = float(np.max(avg_probs))
    else:
        preds = model.predict(Xs)
        # majority vote
        values, counts = np.unique(preds, return_counts=True)
        pred_idx = int(values[np.argmax(counts)])
        confidence = None

    pred_label = IDX_TO_EMOTION[pred_idx]
    return pred_label, confidence

# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser(description="Lightweight Emotion Detection Training")
    parser.add_argument("--dataset_path", type=str,
                        default=r"C:\\Users\\mishr\\Downloads\\Crema-D Subset",
                        help="Path to CREMA-D dataset folder with videos (.mp4/.avi/.flv)")
    parser.add_argument("--feature", type=str, choices=["lbp", "hog"], default="lbp",
                        help="Feature type to extract")
    parser.add_argument("--model", type=str, choices=["lr", "svm", "rf"], default="lr",
                        help="Classifier to train")
    parser.add_argument("--augment", action="store_true", help="Apply lightweight augmentation")
    parser.add_argument("--max_frames_per_video", type=int, default=3,
                        help="Number of frames to sample per video")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--predict_video", type=str, default=None,
                        help="Path to a single video file to predict using saved model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Optional explicit model path to load for prediction")
    parser.add_argument("--scaler_path", type=str, default=None,
                        help="Optional explicit scaler path to load for prediction")
    parser.add_argument("--include_md", action="store_true", help="Include medium intensity clips")
    parser.add_argument("--img_size", type=int, default=48, help="Square image size for preprocessing")
    parser.add_argument("--hog_ppc", type=int, default=8, help="HOG pixels_per_cell (square)")
    parser.add_argument("--hog_cpb", type=int, default=2, help="HOG cells_per_block (square)")
    parser.add_argument("--svm_c", type=float, default=10.0, help="SVM C parameter")
    parser.add_argument("--svm_gamma", type=str, default='scale', help="SVM gamma ('scale','auto' or float)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # If predicting a single video, load saved model/scaler and exit
    if args.predict_video is not None:
        model_path = args.model_path or os.path.join(args.output_dir, f"model_{args.feature}_{args.model}.joblib")
        scaler_path = args.scaler_path or os.path.join(args.output_dir, f"scaler_{args.feature}.joblib")
        try:
            label, conf = predict_video(args.predict_video, args.feature, model_path, scaler_path,
                                        img_size=args.img_size, hog_ppc=args.hog_ppc, hog_cpb=args.hog_cpb)
            if conf is not None:
                logger.info(f"Prediction for {args.predict_video}: {label} (confidence {conf:.3f})")
            else:
                logger.info(f"Prediction for {args.predict_video}: {label}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            sys.exit(1)
        sys.exit(0)

    logger.info("Building dataset...")
    X, y = build_dataset(
        dataset_path=args.dataset_path,
        feature_type=args.feature,
        max_frames_per_video=args.max_frames_per_video,
        augment=args.augment,
        save_aug_example_dir=os.path.join(args.output_dir, "aug_examples") if args.augment else None,
        include_md=args.include_md,
        img_size=args.img_size,
        hog_pixels_per_cell=args.hog_ppc,
        hog_cells_per_block=args.hog_cpb
    )

    if X.shape[0] < 10:
        logger.warning("Very few samples were built. Consider increasing max_frames_per_video or checking filters.")

    logger.info("Training and evaluating model...")
    # parse gamma as float if numeric
    gamma_value = args.svm_gamma
    try:
        if args.svm_gamma not in ('scale', 'auto'):
            gamma_value = float(args.svm_gamma)
    except Exception:
        gamma_value = 'scale'

    results = train_and_evaluate(
        X, y,
        model_type=args.model,
        test_size=args.test_size,
        svm_C=args.svm_c,
        svm_gamma=gamma_value
    )

    # Save scaler and model using joblib
    try:
        joblib.dump(results['scaler'], os.path.join(args.output_dir, f"scaler_{args.feature}.joblib"))
        joblib.dump(results['model'], os.path.join(args.output_dir, f"model_{args.feature}_{args.model}.joblib"))
        logger.info("Saved model and scaler to outputs/")
    except Exception as e:
        logger.warning(f"Could not save model: {e}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
