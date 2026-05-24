"""Visualization utilities for landmarks and predictions."""

import cv2
import numpy as np


class LandmarkVisualizer:
    """Visualize hand landmarks."""
    
    # MediaPipe hand connections
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)             # Palm
    ]
    
    LANDMARK_NAMES = [
        'Wrist',
        'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_TIP',
        'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_TIP',
        'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_TIP',
        'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_TIP',
        'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_TIP'
    ]
    
    @staticmethod
    def draw_landmarks(frame, landmarks, draw_connections=True, draw_labels=False):
        """
        Draw landmarks on frame.
        
        Args:
            frame: Input frame
            landmarks: Landmarks array of shape (63,)
            draw_connections: Whether to draw connections
            draw_labels: Whether to draw landmark labels
        
        Returns:
            Frame with landmarks drawn
        """
        h, w = frame.shape[:2]
        frame_copy = frame.copy()
        
        # Reshape landmarks
        landmarks_2d = landmarks.reshape(21, 3)
        
        # Draw connections
        if draw_connections:
            for start, end in LandmarkVisualizer.CONNECTIONS:
                x1 = int(landmarks_2d[start, 0] * w)
                y1 = int(landmarks_2d[start, 1] * h)
                x2 = int(landmarks_2d[end, 0] * w)
                y2 = int(landmarks_2d[end, 1] * h)
                
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks
        for i in range(21):
            x = int(landmarks_2d[i, 0] * w)
            y = int(landmarks_2d[i, 1] * h)
            
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(frame_copy, (x, y), 6, (255, 0, 0), 2)
                
                if draw_labels:
                    cv2.putText(frame_copy, str(i), (x + 5, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        return frame_copy
    
    @staticmethod
    def draw_with_depth(frame, landmarks, normalize_z=True):
        """
        Draw landmarks with depth-based coloring.
        
        Args:
            frame: Input frame
            landmarks: Landmarks array
            normalize_z: Whether to normalize z coordinate
        
        Returns:
            Frame with depth-colored landmarks
        """
        h, w = frame.shape[:2]
        frame_copy = frame.copy()
        
        landmarks_2d = landmarks.reshape(21, 3)
        
        # Normalize z if needed
        z_values = landmarks_2d[:, 2]
        if normalize_z:
            z_min, z_max = z_values.min(), z_values.max()
            z_values = (z_values - z_min) / (z_max - z_min + 1e-6)
        
        # Draw with depth coloring
        for i in range(21):
            x = int(landmarks_2d[i, 0] * w)
            y = int(landmarks_2d[i, 1] * h)
            z = z_values[i]
            
            # Color based on depth
            if z < 0.33:
                color = (255, 0, 0)  # Blue (far)
            elif z < 0.67:
                color = (0, 255, 0)  # Green (mid)
            else:
                color = (0, 0, 255)  # Red (close)
            
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame_copy, (x, y), 5, color, -1)
                cv2.circle(frame_copy, (x, y), 6, (255, 255, 255), 2)
        
        return frame_copy


class PredictionVisualizer:
    """Visualize predictions."""
    
    @staticmethod
    def draw_prediction(frame, class_id, confidence, class_names=None):
        """
        Draw prediction on frame.
        
        Args:
            frame: Input frame
            class_id: Predicted class ID
            confidence: Confidence score
            class_names: Optional class names
        
        Returns:
            Frame with prediction drawn
        """
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # Prepare text
        if class_names:
            text = f"{class_names[class_id]}: {confidence:.2f}"
        else:
            text = f"Class {class_id}: {confidence:.2f}"
        
        # Draw background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(frame_copy, (10, 10), (10 + text_size[0], 40 + text_size[1]),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame_copy, text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame_copy
    
    @staticmethod
    def draw_top_k(frame, predictions, class_names=None, k=5):
        """
        Draw top-K predictions on frame.
        
        Args:
            frame: Input frame
            predictions: Predictions array
            class_names: Optional class names
            k: Number of top predictions
        
        Returns:
            Frame with predictions drawn
        """
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # Get top-K
        top_k_idx = np.argsort(predictions)[-k:][::-1]
        
        # Draw predictions
        y_offset = 30
        for i, idx in enumerate(top_k_idx):
            if class_names:
                text = f"{i+1}. {class_names[idx]}: {predictions[idx]:.3f}"
            else:
                text = f"{i+1}. Class {idx}: {predictions[idx]:.3f}"
            
            cv2.putText(frame_copy, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25
        
        return frame_copy


def draw_info(frame, text, position=(10, 30), color=(0, 255, 0)):
    """Draw text info on frame."""
    cv2.putText(frame, text, position,
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def draw_fps(frame, fps, position=(10, 30)):
    """Draw FPS on frame."""
    text = f"FPS: {fps:.1f}"
    return draw_info(frame, text, position)
