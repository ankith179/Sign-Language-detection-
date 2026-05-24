"""Video preprocessing and frame extraction utilities."""

import cv2
import numpy as np


class VideoPreprocessor:
    """Preprocess video files."""
    
    def __init__(self, target_fps=30, target_size=(224, 224)):
        """
        Initialize preprocessor.
        
        Args:
            target_fps: Target frames per second
            target_size: Target frame size
        """
        self.target_fps = target_fps
        self.target_size = target_size
    
    def extract_frames(self, video_path, num_frames=16):
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
        
        Returns:
            Array of shape (num_frames, height, width, 3)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to extract
        if total_frames == 0:
            cap.release()
            return None
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        frame_idx = 0
        read_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if read_idx in frame_indices:
                # Resize frame
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
            
            read_idx += 1
            
            if len(frames) == num_frames:
                break
        
        cap.release()
        
        if len(frames) < num_frames:
            # Pad with last frame if needed
            if frames:
                last_frame = frames[-1]
                while len(frames) < num_frames:
                    frames.append(last_frame)
        
        return np.array(frames, dtype=np.float32) / 255.0
    
    def get_video_info(self, video_path):
        """Get video information."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        info = {
            'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info


def resize_frame(frame, size=(224, 224)):
    """Resize frame to target size."""
    return cv2.resize(frame, size)


def normalize_frame(frame):
    """Normalize frame to [0, 1]."""
    return frame.astype(np.float32) / 255.0


def denormalize_frame(frame):
    """Denormalize frame from [0, 1] to [0, 255]."""
    return (frame * 255).astype(np.uint8)


def convert_color(frame, color_space='RGB'):
    """Convert frame color space."""
    if color_space.upper() == 'RGB':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif color_space.upper() == 'GRAY':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif color_space.upper() == 'HSV':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    else:
        return frame


def apply_gaussian_blur(frame, kernel_size=5):
    """Apply Gaussian blur to frame."""
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def apply_histogram_equalization(frame):
    """Apply histogram equalization."""
    if len(frame.shape) == 3:
        # Convert to HSV, equalize V channel, convert back
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        return cv2.equalizeHist(frame)
