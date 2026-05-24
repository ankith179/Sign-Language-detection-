"""Prediction and inference utilities."""

import numpy as np
import tensorflow as tf


class SignLanguagePredictor:
    """Make predictions with trained model."""
    
    def __init__(self, model_path, num_classes=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            num_classes: Number of gesture classes
        """
        self.model = tf.keras.models.load_model(model_path)
        self.num_classes = num_classes or self.model.output_shape[-1]
    
    def predict(self, frames, batch_size=1):
        """
        Make predictions on frames.
        
        Args:
            frames: Input frames of shape (N, height, width, 3) or (1, frames, height, width, 3)
            batch_size: Batch size for prediction
        
        Returns:
            Prediction array
        """
        # Ensure frames are in correct shape
        if len(frames.shape) == 4:
            frames = np.expand_dims(frames, axis=0)
        
        # Make prediction
        predictions = self.model.predict(frames, batch_size=batch_size)
        
        return predictions
    
    def predict_single(self, frames):
        """Predict on single sample."""
        predictions = self.predict(frames)
        return predictions[0]
    
    def get_predicted_class(self, frames, return_confidence=False):
        """
        Get predicted class for frames.
        
        Args:
            frames: Input frames
            return_confidence: Whether to return confidence score
        
        Returns:
            Predicted class ID or (class ID, confidence)
        """
        predictions = self.predict_single(frames)
        class_id = np.argmax(predictions)
        confidence = predictions[class_id]
        
        if return_confidence:
            return class_id, float(confidence)
        else:
            return class_id
    
    def get_top_k(self, frames, k=5):
        """
        Get top-K predictions.
        
        Args:
            frames: Input frames
            k: Number of top predictions
        
        Returns:
            List of (class_id, confidence) tuples
        """
        predictions = self.predict_single(frames)
        top_k_indices = np.argsort(predictions)[-k:][::-1]
        
        results = [(int(idx), float(predictions[idx])) for idx in top_k_indices]
        
        return results


class BatchPredictor:
    """Batch prediction utilities."""
    
    def __init__(self, model_path):
        """Initialize batch predictor."""
        self.model = tf.keras.models.load_model(model_path)
    
    def predict_batch(self, batch_frames, batch_size=16):
        """
        Predict on batch of frames.
        
        Args:
            batch_frames: Batch of frames
            batch_size: Batch size
        
        Returns:
            Predictions array
        """
        return self.model.predict(batch_frames, batch_size=batch_size)
    
    def predict_generator(self, data_generator, steps=None):
        """
        Predict using data generator.
        
        Args:
            data_generator: Data generator
            steps: Number of steps
        
        Returns:
            Predictions array
        """
        return self.model.predict(data_generator, steps=steps)


class ConfidenceFilter:
    """Filter predictions by confidence."""
    
    @staticmethod
    def filter(predictions, threshold=0.5):
        """
        Filter predictions by confidence threshold.
        
        Args:
            predictions: Predictions array
            threshold: Confidence threshold
        
        Returns:
            Filtered class IDs, or -1 for below threshold
        """
        class_ids = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        filtered_classes = np.where(
            confidences >= threshold,
            class_ids,
            -1
        )
        
        return filtered_classes, confidences
    
    @staticmethod
    def smooth(predictions, window_size=3):
        """
        Smooth predictions over time.
        
        Args:
            predictions: Sequence of predictions
            window_size: Smoothing window size
        
        Returns:
            Smoothed predictions
        """
        from scipy.ndimage import uniform_filter1d
        
        if len(predictions) < window_size:
            return predictions
        
        smoothed = np.zeros_like(predictions)
        
        for i in range(predictions.shape[1]):
            smoothed[:, i] = uniform_filter1d(
                predictions[:, i],
                size=window_size,
                mode='nearest'
            )
        
        return smoothed


def ensemble_predict(models, frames):
    """
    Ensemble predictions from multiple models.
    
    Args:
        models: List of models
        frames: Input frames
    
    Returns:
        Ensemble predictions
    """
    predictions = []
    
    for model in models:
        pred = model.predict(frames)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    
    return ensemble_pred
