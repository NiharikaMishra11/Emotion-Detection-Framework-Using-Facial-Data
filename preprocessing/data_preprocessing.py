"""
Data Preprocessing Module for Emotion Detection Framework

This module handles:
- Video frame extraction
- Image resizing and preprocessing
- Grayscale conversion for LBP features
- Color preservation for CNN features
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles preprocessing of video data for emotion detection.
    
    Features:
    - Extract frames from videos at specified intervals
    - Resize frames to target dimensions
    - Convert to grayscale for LBP features
    - Preserve color for CNN features
    - Filter by intensity levels (High only, except Neutral)
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (128, 128),
                 frame_interval: int = 1,
                 emotions: List[str] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            target_size: Target dimensions for resized frames (height, width)
            frame_interval: Extract every nth frame (1 = every frame, 2 = every 2nd frame)
            emotions: List of emotions to process
        """
        self.target_size = target_size
        self.frame_interval = frame_interval
        
        # Map CREMA-D emotion codes to full names
        self.emotion_mapping = {
            'ANG': 'Angry',
            'DIS': 'Disgust', 
            'FEA': 'Fear',
            'HAP': 'Happy',
            'NEU': 'Neutral',
            'SAD': 'Sad'
        }
        
        self.emotions = emotions or list(self.emotion_mapping.values())
        
        # Create output directories
        self.processed_dir = Path('data/processed')
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DataPreprocessor with target size: {target_size}")
    
    def extract_frames_from_video(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of extracted frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return frames
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract every nth frame based on frame_interval
            if frame_count % self.frame_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def preprocess_frame(self, frame: np.ndarray, grayscale: bool = False) -> np.ndarray:
        """
        Preprocess a single frame.
        
        Args:
            frame: Input frame as numpy array
            grayscale: Whether to convert to grayscale
            
        Returns:
            Preprocessed frame
        """
        # Resize frame
        resized = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
        
        # Convert to grayscale if requested
        if grayscale:
            if len(resized.shape) == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        return resized
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse CREMA-D filename format: ActorID_SentenceID_Emotion_Intensity.mp4
        
        Args:
            filename: Video filename
            
        Returns:
            Dictionary with parsed components
        """
        parts = filename.replace('.mp4', '').replace('.flv', '').replace('.avi', '').split('_')
        if len(parts) != 4:
            return {'error': f'Invalid filename format: {filename}'}
        
        return {
            'actor_id': parts[0],
            'sentence_id': parts[1], 
            'emotion_code': parts[2],
            'intensity': parts[3]
        }
    
    def filter_by_intensity(self, filename: str) -> bool:
        """
        Filter videos by intensity level.
        Keep only High intensity clips (except Neutral which includes all).
        
        Args:
            filename: Video filename
            
        Returns:
            True if video should be included, False otherwise
        """
        parsed = self.parse_filename(filename)
        if 'error' in parsed:
            return False
        
        emotion_code = parsed['emotion_code']
        intensity = parsed['intensity']
        
        # For Neutral emotion, include all intensities
        if emotion_code == 'NEU':
            return True
        
        # For other emotions, only include High intensity (HI)
        return intensity == 'HI'
    
    def process_video(self, video_path: str, emotion: str, output_dir: str) -> Dict:
        """
        Process a single video file and save frames.
        
        Args:
            video_path: Path to the video file
            emotion: Emotion label
            output_dir: Output directory for processed frames
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'video_path': video_path,
            'emotion': emotion,
            'frames_extracted': 0,
            'frames_saved': 0,
            'error': None
        }
        
        try:
            # Extract frames
            frames = self.extract_frames_from_video(video_path)
            stats['frames_extracted'] = len(frames)
            
            if not frames:
                stats['error'] = "No frames extracted"
                return stats
            
            # Create output directories
            emotion_dir = Path(output_dir) / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)
            
            color_dir = emotion_dir / 'color'
            grayscale_dir = emotion_dir / 'grayscale'
            color_dir.mkdir(exist_ok=True)
            grayscale_dir.mkdir(exist_ok=True)
            
            # Process and save frames
            video_name = Path(video_path).stem
            for i, frame in enumerate(frames):
                # Save color frame
                color_frame = self.preprocess_frame(frame, grayscale=False)
                color_path = color_dir / f"{video_name}_frame_{i:04d}.jpg"
                cv2.imwrite(str(color_path), color_frame)
                
                # Save grayscale frame
                gray_frame = self.preprocess_frame(frame, grayscale=True)
                gray_path = grayscale_dir / f"{video_name}_frame_{i:04d}.jpg"
                cv2.imwrite(str(gray_path), gray_frame)
                
                stats['frames_saved'] += 1
            
            logger.info(f"Processed {video_path}: {stats['frames_saved']} frames saved")
            
        except Exception as e:
            stats['error'] = str(e)
            logger.error(f"Error processing {video_path}: {e}")
        
        return stats
    
    def process_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Process the entire CREMA-D dataset.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            DataFrame with processing statistics
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        all_stats = []
        
        # Get all video files in the dataset directory
        video_files = (list(dataset_path.glob("*.mp4")) + 
                      list(dataset_path.glob("*.avi")) + 
                      list(dataset_path.glob("*.flv")))
        logger.info(f"Found {len(video_files)} video files")
        
        for video_file in video_files:
            # Parse filename to get emotion
            parsed = self.parse_filename(video_file.name)
            if 'error' in parsed:
                logger.warning(f"Skipping {video_file.name}: {parsed['error']}")
                continue
            
            emotion_code = parsed['emotion_code']
            if emotion_code not in self.emotion_mapping:
                logger.warning(f"Unknown emotion code: {emotion_code}")
                continue
            
            emotion = self.emotion_mapping[emotion_code]
            
            # Filter by intensity
            if not self.filter_by_intensity(video_file.name):
                logger.info(f"Skipping {video_file.name} (not high intensity)")
                continue
            
            # Process video
            stats = self.process_video(
                str(video_file), 
                emotion, 
                str(self.processed_dir)
            )
            all_stats.append(stats)
        
        # Create summary DataFrame
        df = pd.DataFrame(all_stats)
        
        # Save processing summary
        summary_path = self.processed_dir / 'processing_summary.csv'
        df.to_csv(summary_path, index=False)
        
        logger.info(f"Processing complete. Summary saved to: {summary_path}")
        logger.info(f"Total videos processed: {len(df)}")
        
        if len(df) > 0 and 'frames_saved' in df.columns:
            logger.info(f"Total frames saved: {df['frames_saved'].sum()}")
        else:
            logger.warning("No videos were processed successfully")
        
        return df

def main():
    """Example usage of the DataPreprocessor."""
    preprocessor = DataPreprocessor(
        target_size=(128, 128),
        frame_interval=1  # Extract every frame (1 frame per second equivalent)
    )
    
    # Process the CREMA-D dataset
    dataset_path = r"C:\Users\mishr\Downloads\Crema-D Subset"
    df = preprocessor.process_dataset(dataset_path)
    print(df.head())

if __name__ == "__main__":
    main()
