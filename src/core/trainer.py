"""Training utilities and callbacks."""

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)


def get_callbacks(checkpoint_dir='results/checkpoints', log_dir='results/training_logs'):
    """
    Get training callbacks.
    
    Args:
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
    
    Returns:
        List of callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f'{checkpoint_dir}/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks


def get_optimizer(learning_rate=0.001, optimizer_name='adam'):
    """
    Get optimizer.
    
    Args:
        learning_rate: Learning rate
        optimizer_name: Optimizer name
    
    Returns:
        Optimizer instance
    """
    if optimizer_name.lower() == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def compile_model(model, learning_rate=0.001, loss='categorical_crossentropy', metrics=None):
    """
    Compile model.
    
    Args:
        model: Model to compile
        learning_rate: Learning rate
        loss: Loss function
        metrics: Evaluation metrics
    """
    if metrics is None:
        metrics = ['accuracy']
    
    optimizer = get_optimizer(learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


class TrainingMonitor:
    """Monitor training progress."""
    
    def __init__(self):
        """Initialize monitor."""
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def update(self, logs):
        """Update monitor with new logs."""
        self.history['train_loss'].append(logs.get('loss', 0))
        self.history['train_accuracy'].append(logs.get('accuracy', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['val_accuracy'].append(logs.get('val_accuracy', 0))
    
    def get_best_epoch(self):
        """Get best epoch based on validation loss."""
        if not self.history['val_loss']:
            return 0
        
        best_epoch = self.history['val_loss'].index(min(self.history['val_loss']))
        return best_epoch + 1
    
    def get_summary(self):
        """Get training summary."""
        if not self.history['train_loss']:
            return None
        
        return {
            'best_epoch': self.get_best_epoch(),
            'best_val_loss': min(self.history['val_loss']),
            'best_val_accuracy': max(self.history['val_accuracy']),
            'final_train_loss': self.history['train_loss'][-1],
            'final_train_accuracy': self.history['train_accuracy'][-1]
        }
