"""Dataset loading utilities for WLASL and custom datasets."""

import os
import json
import numpy as np
from pathlib import Path


class WLASLDatasetLoader:
    """Load WLASL dataset."""
    
    def __init__(self, dataset_root):
        """
        Initialize WLASL loader.
        
        Args:
            dataset_root: Root directory of WLASL dataset
        """
        self.dataset_root = dataset_root
        self.metadata_file = os.path.join(dataset_root, 'WLASL_v0.3.json')
        self.videos_dir = os.path.join(dataset_root, 'videos')
        self.metadata = None
    
    def load_metadata(self):
        """Load WLASL metadata."""
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        return self.metadata
    
    def get_gestures(self):
        """Get list of all gestures."""
        if self.metadata is None:
            self.load_metadata()
        
        gestures = [item['gloss'] for item in self.metadata]
        return gestures
    
    def get_videos_for_gesture(self, gesture, limit=None):
        """
        Get video paths for a specific gesture.
        
        Args:
            gesture: Gesture name/gloss
            limit: Maximum number of videos to return
        
        Returns:
            List of video paths
        """
        if self.metadata is None:
            self.load_metadata()
        
        videos = []
        
        for item in self.metadata:
            if item['gloss'] == gesture:
                for instance in item['instances']:
                    video_id = str(instance['video_id']).zfill(5)
                    video_path = os.path.join(self.videos_dir, f'{video_id}.mp4')
                    
                    if os.path.exists(video_path):
                        videos.append(video_path)
                    
                    if limit and len(videos) >= limit:
                        return videos
        
        return videos
    
    def get_dataset_stats(self):
        """Get dataset statistics."""
        if self.metadata is None:
            self.load_metadata()
        
        total_gestures = len(self.metadata)
        total_videos = sum(len(item['instances']) for item in self.metadata)
        
        stats = {
            'total_gestures': total_gestures,
            'total_videos': total_videos,
            'metadata_file': self.metadata_file,
            'videos_dir': self.videos_dir
        }
        
        return stats


class CustomDatasetLoader:
    """Load custom dataset."""
    
    def __init__(self, dataset_root):
        """
        Initialize custom dataset loader.
        
        Args:
            dataset_root: Root directory containing videos
        """
        self.dataset_root = dataset_root
    
    def get_video_files(self, extensions=['.mp4', '.avi', '.mov']):
        """Get all video files in dataset."""
        video_files = []
        
        for ext in extensions:
            pattern = f'**/*{ext}'
            video_files.extend(Path(self.dataset_root).glob(pattern))
        
        return [str(f) for f in video_files]
    
    def get_label_from_path(self, video_path, separator='_'):
        """
        Extract label from video file path.
        
        Args:
            video_path: Path to video file
            separator: Separator between gesture and number
        
        Returns:
            Gesture label
        """
        filename = os.path.basename(video_path)
        filename_without_ext = os.path.splitext(filename)[0]
        
        parts = filename_without_ext.split(separator)
        if len(parts) > 0:
            return parts[0]
        
        return None
    
    def create_dataset_structure(self):
        """Create dataset with label-based directory structure."""
        dataset = {}
        
        for video_file in self.get_video_files():
            label = self.get_label_from_path(video_file)
            
            if label:
                if label not in dataset:
                    dataset[label] = []
                
                dataset[label].append(video_file)
        
        return dataset


def get_dataset_loader(dataset_type='wlasl', dataset_root='data/raw'):
    """
    Get appropriate dataset loader.
    
    Args:
        dataset_type: 'wlasl' or 'custom'
        dataset_root: Root directory of dataset
    
    Returns:
        Dataset loader instance
    """
    if dataset_type.lower() == 'wlasl':
        return WLASLDatasetLoader(dataset_root)
    elif dataset_type.lower() == 'custom':
        return CustomDatasetLoader(dataset_root)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
