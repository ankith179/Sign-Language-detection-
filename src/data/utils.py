"""Data utilities for dataset handling and preprocessing."""

import os
import json
import pickle
import numpy as np
import tensorflow as tf


def load_data_split(data_dir, split='train'):
    """Load dataset split (train/val/test)."""
    split_file = os.path.join(data_dir, f'{split}_data.pkl')
    
    if not os.path.exists(split_file):
        return None, None
    
    with open(split_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y = data['y']
    
    return X, y


def save_data_split(data_dir, X, y, split='train'):
    """Save dataset split."""
    os.makedirs(data_dir, exist_ok=True)
    
    split_file = os.path.join(data_dir, f'{split}_data.pkl')
    
    data = {'X': X, 'y': y}
    
    with open(split_file, 'wb') as f:
        pickle.dump(data, f)


def create_tf_dataset(X, y, batch_size=16, shuffle=True):
    """Create TensorFlow dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def encode_labels(y_train, y_val, y_test):
    """Encode labels to integers."""
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    
    return y_train_encoded, y_val_encoded, y_test_encoded, le


def one_hot_encode(y, num_classes):
    """One-hot encode labels."""
    return tf.keras.utils.to_categorical(y, num_classes)


def load_metadata(metadata_file):
    """Load dataset metadata."""
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def save_metadata(metadata_file, metadata):
    """Save dataset metadata."""
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_class_distribution(y):
    """Get distribution of classes."""
    unique, counts = np.unique(y, return_counts=True)
    
    distribution = {int(u): int(c) for u, c in zip(unique, counts)}
    
    return distribution


def augment_landmarks(landmarks, noise_std=0.01):
    """Add Gaussian noise to landmarks for augmentation."""
    noise = np.random.normal(0, noise_std, landmarks.shape)
    augmented = landmarks + noise
    
    # Clip to valid range
    augmented = np.clip(augmented, -1, 1)
    
    return augmented


def filter_by_confidence(landmarks, confidences, threshold=0.5):
    """Filter landmarks by confidence score."""
    valid_indices = confidences >= threshold
    return landmarks[valid_indices]


def normalize_dataset(X_train, X_val=None, X_test=None):
    """Normalize dataset to [0, 1] range."""
    # Calculate statistics from training set
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    
    # Prevent division by zero
    X_max = np.where(X_max == X_min, 1.0, X_max)
    
    # Normalize training set
    X_train_norm = (X_train - X_min) / (X_max - X_min)
    
    # Normalize validation and test sets
    if X_val is not None:
        X_val_norm = (X_val - X_min) / (X_max - X_min)
    else:
        X_val_norm = None
    
    if X_test is not None:
        X_test_norm = (X_test - X_min) / (X_max - X_min)
    else:
        X_test_norm = None
    
    return X_train_norm, X_val_norm, X_test_norm
