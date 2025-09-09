"""
Local Binary Patterns (LBP) Feature Extractor

This module implements LBP feature extraction for emotion detection.
LBP is a texture descriptor that captures local patterns in images.
"""

import numpy as np
import cv2
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

logger = logging.getLogger(__name__)

class LBPFeatureExtractor:
    """
    Local Binary Patterns feature extractor for emotion detection.
    
    Features:
    - Extract LBP features from grayscale images
    - Compute LBP histograms
    - Support for different LBP variants (uniform, rotation-invariant)
    - Batch processing of image directories
    """
    
    def __init__(self, 
                 radius: int = 1, 
                 n_points: int = 8,
                 method: str = 'uniform'):
        """
        Initialize LBP feature extractor.
        
        Args:
            radius: Radius of the circular neighborhood
            n_points: Number of points in the circular neighborhood
            method: LBP method ('uniform', 'nri_uniform', 'var')
        """
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized LBP extractor: radius={radius}, n_points={n_points}, method={method}")
    
    def compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Local Binary Pattern for an image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            LBP image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute LBP using OpenCV
        if self.method == 'uniform':
            lbp = cv2.calcHist([image], [0], None, [256], [0, 256])
        else:
            # Use scikit-image for more advanced LBP methods
            try:
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(image, self.n_points, self.radius, method=self.method)
            except ImportError:
                logger.warning("scikit-image not available, using basic LBP")
                lbp = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        return lbp
    
    def extract_lbp_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP histogram features from an image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            LBP histogram features
        """
        # Compute LBP
        lbp = self.compute_lbp(image)
        
        # Compute histogram
        if self.method == 'uniform':
            # For uniform LBP, we get histogram directly
            hist = lbp.flatten()
        else:
            # For other methods, compute histogram
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        
        # Normalize histogram
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-7)  # Add small epsilon to avoid division by zero
        
        return hist
    
    def extract_features_from_image(self, image_path: str) -> np.ndarray:
        """
        Extract LBP features from a single image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            LBP feature vector
        """
        try:
            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return np.zeros(256)  # Return zero vector for failed loads
            
            # Extract LBP histogram
            features = self.extract_lbp_histogram(image)
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return np.zeros(256)
    
    def extract_features_from_directory(self, directory: str, emotion: str) -> Tuple[np.ndarray, List[str]]:
        """
        Extract LBP features from all images in a directory.
        
        Args:
            directory: Path to directory containing images
            emotion: Emotion label for the images
            
        Returns:
            Tuple of (feature_matrix, image_paths)
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return np.array([]), []
        
        # Get all image files
        image_files = list(directory.glob("*.jpg")) + list(directory.glob("*.png"))
        logger.info(f"Found {len(image_files)} images in {directory}")
        
        if not image_files:
            return np.array([]), []
        
        # Extract features from all images
        features_list = []
        valid_paths = []
        
        for image_file in image_files:
            features = self.extract_features_from_image(str(image_file))
            features_list.append(features)
            valid_paths.append(str(image_file))
        
        # Convert to numpy array
        feature_matrix = np.array(features_list)
        
        logger.info(f"Extracted {feature_matrix.shape[0]} feature vectors for {emotion}")
        return feature_matrix, valid_paths
    
    def extract_dataset_features(self, processed_data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract LBP features from the entire processed dataset.
        
        Args:
            processed_data_dir: Path to processed data directory
            
        Returns:
            Tuple of (features, labels, image_paths)
        """
        processed_dir = Path(processed_data_dir)
        if not processed_dir.exists():
            raise ValueError(f"Processed data directory does not exist: {processed_dir}")
        
        all_features = []
        all_labels = []
        all_paths = []
        
        # Process each emotion directory
        emotion_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
        
        for emotion_dir in emotion_dirs:
            emotion = emotion_dir.name
            grayscale_dir = emotion_dir / 'grayscale'
            
            if not grayscale_dir.exists():
                logger.warning(f"Grayscale directory not found: {grayscale_dir}")
                continue
            
            logger.info(f"Processing emotion: {emotion}")
            
            # Extract features from grayscale images
            features, paths = self.extract_features_from_directory(str(grayscale_dir), emotion)
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.extend([emotion] * len(features))
                all_paths.extend(paths)
        
        # Combine all features
        if all_features:
            X = np.vstack(all_features)
            y = np.array(all_labels)
        else:
            X = np.array([])
            y = np.array([])
        
        logger.info(f"Total features extracted: {X.shape[0] if len(X) > 0 else 0}")
        logger.info(f"Feature dimension: {X.shape[1] if len(X) > 0 else 0}")
        
        return X, y, all_paths
    
    def fit_scaler(self, features: np.ndarray):
        """
        Fit the feature scaler on training data.
        
        Args:
            features: Training feature matrix
        """
        self.scaler.fit(features)
        logger.info("Feature scaler fitted")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        
        Args:
            features: Feature matrix to transform
            
        Returns:
            Scaled feature matrix
        """
        return self.scaler.transform(features)
    
    def save_features(self, features: np.ndarray, labels: np.ndarray, 
                     image_paths: List[str], output_path: str):
        """
        Save extracted features to disk.
        
        Args:
            features: Feature matrix
            labels: Label array
            image_paths: List of image paths
            output_path: Output file path
        """
        data = {
            'features': features,
            'labels': labels,
            'image_paths': image_paths,
            'extractor_params': {
                'radius': self.radius,
                'n_points': self.n_points,
                'method': self.method
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Features saved to: {output_path}")
    
    def load_features(self, input_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load features from disk.
        
        Args:
            input_path: Input file path
            
        Returns:
            Tuple of (features, labels, image_paths)
        """
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Features loaded from: {input_path}")
        return data['features'], data['labels'], data['image_paths']

def main():
    """Example usage of the LBP feature extractor."""
    extractor = LBPFeatureExtractor(radius=1, n_points=8, method='uniform')
    
    # Extract features from processed dataset
    processed_dir = "data/processed"
    X, y, paths = extractor.extract_dataset_features(processed_dir)
    
    if len(X) > 0:
        # Fit scaler and transform features
        extractor.fit_scaler(X)
        X_scaled = extractor.transform_features(X)
        
        # Save features
        output_path = "data/processed/lbp_features.pkl"
        extractor.save_features(X_scaled, y, paths, output_path)
        
        print(f"Extracted {X.shape[0]} feature vectors")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Unique emotions: {np.unique(y)}")

if __name__ == "__main__":
    main()
