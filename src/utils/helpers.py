"""Helper utilities for the Sign Language Recognition System."""

import os
import numpy as np
import tensorflow as tf


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_gpu(gpu_id=0):
    """
    Setup GPU for training.
    
    Args:
        gpu_id: GPU device ID
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPU found, using CPU")
        return False
    
    try:
        # Set memory growth to prevent OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set visible device
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        print(f"✓ GPU {gpu_id} configured successfully")
        return True
    
    except RuntimeError as e:
        print(f"GPU setup failed: {e}")
        return False


def create_directories(paths):
    """Create necessary directories."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        print("PyYAML not installed, returning empty config")
        return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def save_checkpoint(model, checkpoint_dir, epoch):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.h5')
    model.save(checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path):
    """Load model checkpoint."""
    try:
        model = tf.keras.models.load_model(checkpoint_path)
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def calculate_class_weights(y_train):
    """Calculate class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    return {i: w for i, w in enumerate(weights)}


def normalize_landmarks(landmarks):
    """Normalize landmarks to [0, 1] range."""
    if landmarks is None or len(landmarks) == 0:
        return None
    
    # Normalize x and y to [0, 1]
    landmarks_norm = landmarks.copy()
    
    # Assuming landmarks are in format [x, y, z, x, y, z, ...]
    # x and y are typically already in [0, 1], z is in [-1, 1]
    
    return landmarks_norm


def smooth_predictions(predictions, window_size=3):
    """Smooth predictions over time window."""
    from scipy.ndimage import uniform_filter1d
    
    if len(predictions) < window_size:
        return predictions
    
    smoothed = uniform_filter1d(predictions, size=window_size, mode='nearest')
    return smoothed


def get_top_k_predictions(predictions, k=5):
    """Get top-K predictions with confidence scores."""
    top_k_idx = np.argsort(predictions)[-k:][::-1]
    top_k_conf = predictions[top_k_idx]
    
    return list(zip(top_k_idx, top_k_conf))


def format_time(seconds):
    """Format time in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
