"""
Lightweight Generative AI Augmentation Module

This module provides synthetic data augmentation using simple transformations
to simulate generative AI effects without heavy computational requirements.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import random
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class GenAIAugment:
    """
    Lightweight generative AI augmentation using traditional computer vision techniques.
    
    Features:
    - Horizontal/vertical flips
    - Small rotations
    - Brightness/contrast changes
    - Gaussian noise
    - Color space transformations
    - Simulated emotion intensity variations
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the augmentation module.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info("Initialized GenAIAugment module")
    
    def horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """Apply horizontal flip transformation."""
        return cv2.flip(image, 1)
    
    def vertical_flip(self, image: np.ndarray) -> np.ndarray:
        """Apply vertical flip transformation."""
        return cv2.flip(image, 0)
    
    def rotate(self, image: np.ndarray, angle_range: Tuple[float, float] = (-15, 15)) -> np.ndarray:
        """
        Apply small rotation transformation.
        
        Args:
            image: Input image
            angle_range: Range of rotation angles in degrees
        """
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT_101)
        
        return rotated
    
    def brightness_contrast(self, image: np.ndarray, 
                          brightness_range: Tuple[float, float] = (0.8, 1.2),
                          contrast_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Apply brightness and contrast adjustments.
        
        Args:
            image: Input image
            brightness_range: Range for brightness multiplier
            contrast_range: Range for contrast multiplier
        """
        brightness = random.uniform(brightness_range[0], brightness_range[1])
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        
        # Apply brightness and contrast
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness - 1) * 100)
        
        return adjusted
    
    def gaussian_noise(self, image: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to the image.
        
        Args:
            image: Input image
            noise_factor: Strength of noise (0-1)
        """
        noise = np.random.normal(0, noise_factor * 255, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        
        return noisy_image
    
    def color_jitter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color space transformations to simulate different lighting conditions.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Randomly adjust hue, saturation, and value
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180  # Hue
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)  # Saturation
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)  # Value
        
        # Convert back to BGR
        jittered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return jittered
    
    def emotion_intensity_simulation(self, image: np.ndarray, emotion: str) -> np.ndarray:
        """
        Simulate different emotion intensities using color and contrast adjustments.
        
        Args:
            image: Input image
            emotion: Emotion type for intensity simulation
        """
        if emotion.lower() in ['angry', 'disgust']:
            # Increase red channel for anger/disgust
            image = image.copy()
            image[:, :, 2] = np.clip(image[:, :, 2] * 1.1, 0, 255)
            image = self.brightness_contrast(image, (1.0, 1.1), (1.0, 1.1))
            
        elif emotion.lower() in ['happy', 'fear']:
            # Increase brightness for happy/fear
            image = self.brightness_contrast(image, (1.0, 1.2), (1.0, 1.1))
            
        elif emotion.lower() in ['sad', 'neutral']:
            # Decrease saturation for sad/neutral
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.9, 0, 255)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image
    
    def genai_augment(self, image: np.ndarray, emotion: str = None, 
                     num_augmentations: int = 5) -> List[np.ndarray]:
        """
        Main augmentation function that applies multiple transformations.
        
        Args:
            image: Input image
            emotion: Emotion label for intensity simulation
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented images
        """
        augmented_images = [image.copy()]  # Include original
        
        for _ in range(num_augmentations):
            aug_image = image.copy()
            
            # Apply random transformations
            if random.random() < 0.5:
                aug_image = self.horizontal_flip(aug_image)
            
            if random.random() < 0.3:
                aug_image = self.rotate(aug_image)
            
            if random.random() < 0.7:
                aug_image = self.brightness_contrast(aug_image)
            
            if random.random() < 0.4:
                aug_image = self.gaussian_noise(aug_image)
            
            if random.random() < 0.5:
                aug_image = self.color_jitter(aug_image)
            
            # Apply emotion-specific intensity simulation
            if emotion and random.random() < 0.6:
                aug_image = self.emotion_intensity_simulation(aug_image, emotion)
            
            augmented_images.append(aug_image)
        
        return augmented_images
    
    def augment_dataset(self, input_dir: str, output_dir: str, 
                       emotions: List[str] = None, samples_per_emotion: int = 100):
        """
        Augment an entire dataset directory.
        
        Args:
            input_dir: Input directory containing emotion subdirectories
            output_dir: Output directory for augmented images
            emotions: List of emotions to process
            samples_per_emotion: Number of augmented samples per emotion
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        emotions = emotions or ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
        
        for emotion in emotions:
            emotion_input_dir = input_path / emotion / 'color'
            emotion_output_dir = output_path / emotion
            emotion_output_dir.mkdir(parents=True, exist_ok=True)
            
            if not emotion_input_dir.exists():
                logger.warning(f"Input directory not found: {emotion_input_dir}")
                continue
            
            # Get image files
            image_files = list(emotion_input_dir.glob("*.jpg"))
            if not image_files:
                logger.warning(f"No images found in: {emotion_input_dir}")
                continue
            
            logger.info(f"Augmenting {emotion}: {len(image_files)} images")
            
            # Augment images
            for i, image_file in enumerate(image_files[:samples_per_emotion]):
                try:
                    # Load image
                    image = cv2.imread(str(image_file))
                    if image is None:
                        continue
                    
                    # Generate augmentations
                    augmented_images = self.genai_augment(image, emotion, num_augmentations=3)
                    
                    # Save augmented images
                    base_name = image_file.stem
                    for j, aug_image in enumerate(augmented_images):
                        output_file = emotion_output_dir / f"{base_name}_aug_{j}.jpg"
                        cv2.imwrite(str(output_file), aug_image)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{min(len(image_files), samples_per_emotion)} images")
                
                except Exception as e:
                    logger.error(f"Error processing {image_file}: {e}")
                    continue
        
        logger.info(f"Dataset augmentation completed. Output saved to: {output_path}")
    
    def show_augmentation_examples(self, image_path: str, emotion: str = None, 
                                 save_path: str = None):
        """
        Show before and after examples of augmentation.
        
        Args:
            image_path: Path to input image
            emotion: Emotion label
            save_path: Path to save the visualization
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return
        
        # Generate augmentations
        augmented_images = self.genai_augment(image, emotion, num_augmentations=5)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, aug_image in enumerate(augmented_images):
            if i < len(axes):
                # Convert BGR to RGB for matplotlib
                display_image = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
                axes[i].imshow(display_image)
                axes[i].set_title(f"Augmentation {i}")
                axes[i].axis('off')
        
        plt.suptitle(f"Generative AI Augmentation Examples - {emotion or 'Unknown'}", 
                    fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Augmentation examples saved to: {save_path}")
        
        plt.show()

def main():
    """Example usage of the augmentation module."""
    augmenter = GenAIAugment()
    
    # Example: Augment a single image
    # image_path = "path/to/your/image.jpg"
    # augmented_images = augmenter.genai_augment(cv2.imread(image_path), "Happy")
    
    # Example: Show augmentation examples
    # augmenter.show_augmentation_examples(image_path, "Happy", "augmentation_examples.png")
    
    # Example: Augment entire dataset
    # augmenter.augment_dataset("data/processed", "data/augmented", samples_per_emotion=50)
    
    print("GenAI Augmentation module ready!")

if __name__ == "__main__":
    main()
