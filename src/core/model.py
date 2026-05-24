"""
Sign Language Recognition Using MediaPipe, CNN, and Transformer
Complete Implementation for WLASL Dataset
Author: AI Expert | Date: 2024
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)
import mediapipe as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    'FRAME_SIZE': 16,              # Frames per video
    'IMG_SIZE': (224, 224),        # Frame resolution for CNN
    'LANDMARK_DIM': 63,            # 21 points × 3 (x, y, z)
    'BATCH_SIZE': 16,
    'EPOCHS': 40,
    'LEARNING_RATE': 0.001,
    'SEED': 42,
    'VAL_SPLIT': 0.2,
    'TEST_SPLIT': 0.1,
}

np.random.seed(CONFIG['SEED'])
tf.random.set_seed(CONFIG['SEED'])

# =====================================================================
# PART 1: MEDIAPIPE HAND LANDMARK EXTRACTION & VISUALIZATION
# =====================================================================

class MediaPipeHandDetector:
    """Extract hand landmarks using MediaPipe with visualization support"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def get_landmarks(self, frame):
        """
        Extract landmarks from a frame
        Returns: (landmarks_array, frame_with_landmarks)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks_array = np.zeros(CONFIG['LANDMARK_DIM'])
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract 21 landmarks (x, y, z) = 63 features
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmarks_array[idx * 3] = landmark.x
                    landmarks_array[idx * 3 + 1] = landmark.y
                    landmarks_array[idx * 3 + 2] = landmark.z
                
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        
        return landmarks_array, annotated_frame
    
    def visualize_landmarks(self, frame, landmarks):
        """Create enhanced visualization of landmarks"""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw circles for each landmark
        for i in range(0, CONFIG['LANDMARK_DIM'], 3):
            x = int(landmarks[i] * w)
            y = int(landmarks[i+1] * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(annotated, (x, y), 6, (255, 0, 0), 2)
        
        return annotated

# =====================================================================
# PART 2: VIDEO PREPROCESSING & DATASET LOADING
# =====================================================================

class VideoPreprocessor:
    """Process videos and extract landmark sequences"""
    
    def __init__(self):
        self.detector = MediaPipeHandDetector()
        
    def load_video(self, video_path, num_frames=CONFIG['FRAME_SIZE']):
        """
        Load video and extract frames uniformly
        Returns: landmark_sequence (num_frames, landmark_dim)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return None, None
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        landmarks_sequence = []
        frames_with_landmarks = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # Pad with zeros if frame not available
                landmarks_sequence.append(np.zeros(CONFIG['LANDMARK_DIM']))
                frames_with_landmarks.append(np.zeros_like(frame) if frame is not None else None)
                continue
            
            landmarks, annotated_frame = self.detector.get_landmarks(frame)
            landmarks_sequence.append(landmarks)
            frames_with_landmarks.append(annotated_frame)
        
        cap.release()
        
        return np.array(landmarks_sequence), frames_with_landmarks
    
    def load_frame_sequence(self, video_path, num_frames=CONFIG['FRAME_SIZE']):
        """Load raw frame sequence for CNN input"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return None
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, CONFIG['IMG_SIZE'])
                frame = frame.astype('float32') / 255.0
                frames.append(frame)
            else:
                frames.append(np.zeros((*CONFIG['IMG_SIZE'], 3)))
        
        cap.release()
        return np.array(frames)

# =====================================================================
# PART 3: CNN FEATURE EXTRACTOR
# =====================================================================

class CNNFeatureExtractor(keras.Model):
    """Lightweight CNN for frame-level feature extraction"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2)
        
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2)
        
        self.conv3 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(2)
        
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.3)
    
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        
        x = self.global_pool(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return x

# =====================================================================
# PART 4: TRANSFORMER MODEL
# =====================================================================

class TransformerBlock(layers.Layer):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_hybrid_model(num_classes, input_shape_landmarks=(CONFIG['FRAME_SIZE'], CONFIG['LANDMARK_DIM']),
                       input_shape_frames=(CONFIG['FRAME_SIZE'], *CONFIG['IMG_SIZE'], 3)):
    """
    Hybrid CNN + Transformer Model
    - CNN: Processes frame-level features
    - Transformer: Learns temporal dependencies
    """
    
    # ---- LANDMARK PATHWAY ----
    landmark_input = keras.Input(shape=input_shape_landmarks, name='landmark_input')
    
    # Embed landmarks
    x_landmark = layers.Dense(128, activation='relu')(landmark_input)
    x_landmark = layers.Dropout(0.3)(x_landmark)
    
    # Add positional encoding
    pos_encoding = tf.range(CONFIG['FRAME_SIZE'])[tf.newaxis, :, tf.newaxis]
    pos_encoding = tf.cast(pos_encoding, tf.float32)
    x_landmark = x_landmark + pos_encoding * 0.1
    
    # Transformer blocks
    x_landmark = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, rate=0.1)(x_landmark)
    x_landmark = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, rate=0.1)(x_landmark)
    
    # ---- FRAME PATHWAY (CNN) ----
    frame_input = keras.Input(shape=input_shape_frames, name='frame_input')
    
    # Reshape frames for CNN processing
    x_frame = layers.Reshape((CONFIG['FRAME_SIZE'] * CONFIG['IMG_SIZE'][0], 
                              CONFIG['IMG_SIZE'][1], 3))(frame_input)
    
    # CNN feature extraction
    cnn_extractor = CNNFeatureExtractor()
    cnn_features = []
    for i in range(CONFIG['FRAME_SIZE']):
        frame = frame_input[:, i, :, :, :]
        feat = cnn_extractor(frame)
        cnn_features.append(feat)
    
    x_frame = layers.Stack()(cnn_features)  # (batch, frame_size, 256)
    
    # Transformer on CNN features
    x_frame = TransformerBlock(embed_dim=256, num_heads=4, ff_dim=512, rate=0.1)(x_frame)
    x_frame = TransformerBlock(embed_dim=256, num_heads=4, ff_dim=512, rate=0.1)(x_frame)
    
    # ---- FUSION ----
    # Project landmark features to match frame features
    x_landmark = layers.Dense(256)(x_landmark)
    
    # Concatenate and pool
    x_fusion = layers.Concatenate()([x_landmark, x_frame])
    x_fusion = layers.GlobalAveragePooling1D()(x_fusion)
    
    # ---- CLASSIFICATION HEAD ----
    x = layers.Dense(512, activation='relu')(x_fusion)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=[landmark_input, frame_input], outputs=output)
    return model

# =====================================================================
# PART 5: TRAINING PIPELINE
# =====================================================================

def create_callbacks(model_name='slr_model'):
    """Create training callbacks for optimal performance"""
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=f'./logs/{model_name}'
        )
    ]

def compile_and_train(model, X_train_landmarks, X_train_frames, y_train,
                     X_val_landmarks, X_val_frames, y_val, epochs=CONFIG['EPOCHS']):
    """Compile and train the model"""
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                keras.metrics.Precision(),
                keras.metrics.Recall()]
    )
    
    history = model.fit(
        [X_train_landmarks, X_train_frames],
        y_train,
        validation_data=([X_val_landmarks, X_val_frames], y_val),
        epochs=epochs,
        batch_size=CONFIG['BATCH_SIZE'],
        callbacks=create_callbacks(),
        verbose=1
    )
    
    return history

# =====================================================================
# PART 6: EVALUATION METRICS
# =====================================================================

def evaluate_model(model, X_test_landmarks, X_test_frames, y_test, label_encoder):
    """Comprehensive model evaluation"""
    
    y_pred_probs = model.predict([X_test_landmarks, X_test_frames])
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test_labels, y_pred)
    precision = precision_score(y_test_labels, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_labels, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_labels, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60)
    
    # Classification Report
    class_names = label_encoder.classes_
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test_labels, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_labels, y_pred)
    return accuracy, precision, recall, f1, cm, y_pred_probs

def plot_confusion_matrix(cm, label_encoder):
    """Visualize confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Loss', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[2].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[2].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    axes[2].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[2].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[2].set_title('Precision & Recall', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# =====================================================================
# PART 7: REAL-TIME PREDICTION SYSTEM
# =====================================================================

class RealTimeSignLanguageRecognizer:
    """Real-time sign language recognition from webcam"""
    
    def __init__(self, model, label_encoder, preprocessor):
        self.model = model
        self.label_encoder = label_encoder
        self.preprocessor = preprocessor
        self.detector = MediaPipeHandDetector()
        self.frame_buffer = []
        self.landmark_buffer = []
        self.confidence_threshold = 0.7
        
    def predict_gesture(self, frames_sequence):
        """Predict gesture from frame sequence"""
        if len(frames_sequence) < CONFIG['FRAME_SIZE']:
            return None, 0
        
        frames = np.array(frames_sequence[-CONFIG['FRAME_SIZE']:])
        landmarks = np.array(self.landmark_buffer[-CONFIG['FRAME_SIZE']:])
        
        # Pad if necessary
        if len(frames) < CONFIG['FRAME_SIZE']:
            pad_frames = np.zeros((CONFIG['FRAME_SIZE'] - len(frames), *CONFIG['IMG_SIZE'], 3))
            frames = np.concatenate([pad_frames, frames])
        
        if len(landmarks) < CONFIG['FRAME_SIZE']:
            pad_landmarks = np.zeros((CONFIG['FRAME_SIZE'] - len(landmarks), CONFIG['LANDMARK_DIM']))
            landmarks = np.concatenate([pad_landmarks, landmarks])
        
        # Add batch dimension
        frames = np.expand_dims(frames, axis=0)
        landmarks = np.expand_dims(landmarks, axis=0)
        
        # Predict
        probs = self.model.predict([landmarks, frames], verbose=0)
        pred_idx = np.argmax(probs[0])
        confidence = probs[0][pred_idx]
        
        if confidence >= self.confidence_threshold:
            predicted_gesture = self.label_encoder.inverse_transform([pred_idx])[0]
            return predicted_gesture, float(confidence)
        
        return None, float(confidence)
    
    def run(self, video_source=0):
        """Run real-time recognition"""
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*60)
        print("REAL-TIME SIGN LANGUAGE RECOGNITION")
        print("="*60)
        print(f"Frame buffer size: {CONFIG['FRAME_SIZE']}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("Press 'q' to exit | 'r' to reset buffer")
        print("="*60)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Resize frame for processing
            process_frame = cv2.resize(frame, CONFIG['IMG_SIZE'])
            process_frame_norm = process_frame.astype('float32') / 255.0
            
            # Extract landmarks
            landmarks, annotated_frame = self.detector.get_landmarks(frame)
            
            # Add to buffers
            self.frame_buffer.append(process_frame_norm)
            self.landmark_buffer.append(landmarks)
            
            # Keep buffer size
            if len(self.frame_buffer) > CONFIG['FRAME_SIZE']:
                self.frame_buffer.pop(0)
                self.landmark_buffer.pop(0)
            
            # Predict every 5 frames for efficiency
            if frame_count % 5 == 0 and len(self.frame_buffer) == CONFIG['FRAME_SIZE']:
                gesture, confidence = self.predict_gesture(self.frame_buffer)
            
            frame_count += 1
            
            # Display annotated frame
            cv2.putText(frame, f'Frames in buffer: {len(self.frame_buffer)}/{CONFIG["FRAME_SIZE"]}',
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if 'gesture' in locals() and gesture:
                color = (0, 255, 0) if confidence >= 0.8 else (0, 165, 255)
                cv2.rectangle(frame, (20, 100), (500, 200), color, -1)
                cv2.putText(frame, f'Gesture: {gesture}', (40, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cv2.putText(frame, f'Confidence: {confidence:.2f}', (40, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Sign Language Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.frame_buffer.clear()
                self.landmark_buffer.clear()
                print("Buffer reset!")
        
        cap.release()
        cv2.destroyAllWindows()

# =====================================================================
# PART 8: VISUALIZATION OF MEDIAPIPE LANDMARKS
# =====================================================================

def visualize_mediapipe_landmarks(video_path, num_frames=5):
    """Visualize MediaPipe 21 landmarks detection"""
    preprocessor = VideoPreprocessor()
    landmarks_seq, frames_with_landmarks = preprocessor.load_video(video_path, num_frames=num_frames)
    
    if landmarks_seq is None:
        print(f"Error loading video: {video_path}")
        return
    
    fig, axes = plt.subplots(num_frames, 2, figsize=(15, num_frames*4))
    if num_frames == 1:
        axes = axes.reshape(1, -1)
    
    for i, (landmarks, frame) in enumerate(zip(landmarks_seq, frames_with_landmarks)):
        # Left: Frame with landmarks
        if frame is not None:
            axes[i, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Frame {i+1} - Detected Landmarks', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Right: Landmark coordinates visualization
        landmark_names = [
            'Wrist', 'Thumb CMC', 'Thumb MCP', 'Thumb IP', 'Thumb Tip',
            'Index MCP', 'Index PIP', 'Index DIP', 'Index Tip',
            'Middle MCP', 'Middle PIP', 'Middle DIP', 'Middle Tip',
            'Ring MCP', 'Ring PIP', 'Ring DIP', 'Ring Tip',
            'Pinky MCP', 'Pinky PIP', 'Pinky DIP', 'Pinky Tip'
        ]
        
        # Create scatter plot of landmarks
        x_coords = landmarks[0::3]
        y_coords = landmarks[1::3]
        z_coords = landmarks[2::3]
        
        axes[i, 1].scatter(x_coords, y_coords, s=50, c=z_coords, cmap='viridis', edgecolors='black')
        axes[i, 1].set_xlim([0, 1])
        axes[i, 1].set_ylim([1, 0])
        axes[i, 1].set_xlabel('X Coordinate')
        axes[i, 1].set_ylabel('Y Coordinate')
        axes[i, 1].set_title(f'Frame {i+1} - Landmark Coordinates', fontweight='bold')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(axes[i, 1].collections[0], ax=axes[i, 1])
        cbar.set_label('Z (Depth)')
    
    plt.tight_layout()
    plt.savefig('mediapipe_landmarks_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Visualization saved: mediapipe_landmarks_visualization.png")

# =====================================================================
# MAIN EXECUTION FUNCTION
# =====================================================================

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*60)
    print("SIGN LANGUAGE RECOGNITION SYSTEM - HYBRID ARCHITECTURE")
    print("="*60)
    
    # NOTE: For actual implementation, load WLASL dataset
    # This is a demonstration with synthetic data
    
    # Step 1: Create sample data (replace with WLASL dataset)
    print("\n[1] Preparing Data...")
    num_classes = 10  # Replace with actual WLASL classes
    num_samples = 100  # Replace with actual dataset size
    
    X_landmarks = np.random.randn(num_samples, CONFIG['FRAME_SIZE'], CONFIG['LANDMARK_DIM'])
    X_frames = np.random.randn(num_samples, CONFIG['FRAME_SIZE'], *CONFIG['IMG_SIZE'], 3)
    y = keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)
    
    # Label encoder for classes
    label_encoder = LabelEncoder()
    label_encoder.fit([f'Gesture_{i}' for i in range(num_classes)])
    
    # Split data
    split_val = int(0.8 * num_samples)
    split_test = int(0.9 * num_samples)
    
    X_train_landmarks, X_val_landmarks = X_landmarks[:split_val], X_landmarks[split_val:split_test]
    X_test_landmarks = X_landmarks[split_test:]
    
    X_train_frames, X_val_frames = X_frames[:split_val], X_frames[split_val:split_test]
    X_test_frames = X_frames[split_test:]
    
    y_train, y_val = y[:split_val], y[split_val:split_test]
    y_test = y[split_test:]
    
    print(f"  Train: {X_train_landmarks.shape[0]} samples")
    print(f"  Val:   {X_val_landmarks.shape[0]} samples")
    print(f"  Test:  {X_test_landmarks.shape[0]} samples")
    
    # Step 2: Build model
    print("\n[2] Building Hybrid CNN+Transformer Model...")
    model = create_hybrid_model(num_classes)
    model.summary()
    
    # Step 3: Train model
    print("\n[3] Training Model...")
    history = compile_and_train(
        model,
        X_train_landmarks, X_train_frames, y_train,
        X_val_landmarks, X_val_frames, y_val,
        epochs=CONFIG['EPOCHS']
    )
    
    # Step 4: Evaluate model
    print("\n[4] Evaluating Model...")
    accuracy, precision, recall, f1, cm, y_pred_probs = evaluate_model(
        model, X_test_landmarks, X_test_frames, y_test, label_encoder
    )
    
    # Step 5: Visualizations
    print("\n[5] Creating Visualizations...")
    plot_training_history(history)
    plot_confusion_matrix(cm, label_encoder)
    
    # Step 6: Save model
    print("\n[6] Saving Model...")
    model.save('sign_language_recognition_model.h5')
    print("✓ Model saved: sign_language_recognition_model.h5")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return model, label_encoder

if __name__ == "__main__":
    model, label_encoder = main()
    
    # Uncomment for real-time inference:
    # recognizer = RealTimeSignLanguageRecognizer(model, label_encoder, VideoPreprocessor())
    # recognizer.run(video_source=0)
