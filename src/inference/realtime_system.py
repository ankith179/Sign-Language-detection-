"""
Real-Time Sign Language Recognition with Advanced MediaPipe Visualization
Complete Real-time System with 21-point Hand Landmark Detection
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

# =====================================================================
# ADVANCED MEDIAPIPE VISUALIZER WITH 21 LANDMARKS
# =====================================================================

class AdvancedMediaPipeVisualizer:
    """
    Advanced MediaPipe visualizer with 21-point hand landmarks
    Shows connection lines, depth information, and confidence scores
    """
    
    # MediaPipe hand landmark mapping
    LANDMARK_NAMES = [
        'Wrist (0)',
        'Thumb_CMC (1)', 'Thumb_MCP (2)', 'Thumb_IP (3)', 'Thumb_Tip (4)',
        'Index_MCP (5)', 'Index_PIP (6)', 'Index_DIP (7)', 'Index_Tip (8)',
        'Middle_MCP (9)', 'Middle_PIP (10)', 'Middle_DIP (11)', 'Middle_Tip (12)',
        'Ring_MCP (13)', 'Ring_PIP (14)', 'Ring_DIP (15)', 'Ring_Tip (16)',
        'Pinky_MCP (17)', 'Pinky_PIP (18)', 'Pinky_DIP (19)', 'Pinky_Tip (20)'
    ]
    
    # Hand connections (21 points)
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17)
    ]
    
    def __init__(self, confidence_threshold=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=0.5
        )
        self.confidence_threshold = confidence_threshold
        
    def extract_and_visualize(self, frame, draw_labels=True, draw_connections=True):
        """
        Extract landmarks and create visualization
        Returns: landmarks_array, annotated_frame, hand_info
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        h, w = frame.shape[:2]
        annotated = frame.copy()
        landmarks_list = []
        hand_info = {'detected': False, 'num_hands': 0, 'landmarks': []}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_info['detected'] = True
            hand_info['num_hands'] = len(results.multi_hand_landmarks)
            
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            )):
                # Extract 21 landmarks (x, y, z)
                hand_landmarks_array = np.zeros(63)
                landmark_coords = []
                
                for lm_idx, landmark in enumerate(hand_landmarks.landmark):
                    # Store in array
                    hand_landmarks_array[lm_idx * 3] = landmark.x
                    hand_landmarks_array[lm_idx * 3 + 1] = landmark.y
                    hand_landmarks_array[lm_idx * 3 + 2] = landmark.z
                    
                    # Store coordinates for visualization
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    landmark_coords.append({
                        'index': lm_idx,
                        'name': self.LANDMARK_NAMES[lm_idx],
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'px': px,
                        'py': py
                    })
                
                landmarks_list.append(hand_landmarks_array)
                hand_info['landmarks'].append({
                    'hand': handedness.classification[0].label,
                    'confidence': handedness.classification[0].score,
                    'coords': landmark_coords
                })
                
                # Draw connections
                if draw_connections:
                    self._draw_connections(annotated, landmark_coords, hand_idx)
                
                # Draw landmarks (points)
                for lm in landmark_coords:
                    # Color by depth (z-coordinate)
                    depth_normalized = (lm['z'] + 1.0) / 2.0  # Normalize -1 to 1 as 0 to 1
                    color = self._get_depth_color(depth_normalized)
                    
                    # Draw circle
                    cv2.circle(annotated, (lm['px'], lm['py']), 6, color, -1)
                    cv2.circle(annotated, (lm['px'], lm['py']), 7, (0, 255, 255), 2)
                    
                    # Draw label if enabled
                    if draw_labels:
                        label = f"{lm['index']}"
                        cv2.putText(annotated, label, 
                                  (lm['px'] + 10, lm['py']), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                                  (255, 255, 255), 1, cv2.LINE_AA)
        
        return np.array(landmarks_list) if landmarks_list else np.zeros((0, 63)), annotated, hand_info
    
    def _draw_connections(self, frame, landmark_coords, hand_idx):
        """Draw connection lines between landmarks"""
        h, w = frame.shape[:2]
        
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmark_coords[start_idx]
            end = landmark_coords[end_idx]
            
            # Thickness based on depth
            thickness = max(1, int(3 * ((start['z'] + end['z']) / 2 + 1.0) / 2.0))
            
            cv2.line(frame, 
                    (start['px'], start['py']),
                    (end['px'], end['py']),
                    (0, 255, 0), thickness)
    
    def _get_depth_color(self, depth):
        """Get color based on depth (z-coordinate)"""
        # Blue (far) -> Green (mid) -> Red (close)
        if depth < 0.33:
            return (255, 0, 0)  # Blue
        elif depth < 0.66:
            return (0, 255, 0)  # Green
        else:
            return (0, 0, 255)  # Red
    
    def draw_landmark_info(self, frame, hand_info, position=(10, 30)):
        """Draw hand detection information on frame"""
        y_offset = position[1]
        
        if hand_info['detected']:
            # Detection status
            cv2.putText(frame, f"Hands detected: {hand_info['num_hands']}", 
                       (position[0], y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                       (0, 255, 0), 2)
            y_offset += 35
            
            # Hand information
            for hand_idx, hand in enumerate(hand_info['landmarks']):
                text = f"{hand['hand']} Hand - Confidence: {hand['confidence']:.2f}"
                cv2.putText(frame, text, 
                           (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0), 1)
                y_offset += 25
        else:
            cv2.putText(frame, "No hands detected", 
                       (position[0], y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 0, 255), 2)
    
    def create_landmark_visualization(self, hand_info, frame_width=400, frame_height=600):
        """Create detailed 2D visualization of landmarks"""
        vis_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
        
        if not hand_info['landmarks']:
            cv2.putText(vis_frame, "No hands detected",
                       (50, frame_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return vis_frame
        
        hand = hand_info['landmarks'][0]
        
        # Draw title
        title = f"{hand['hand']} Hand - 21 Landmarks"
        cv2.putText(vis_frame, title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Scale landmarks to visualization frame
        y_text_start = 60
        for idx, lm in enumerate(hand['coords']):
            text = f"{lm['index']:2d}: {lm['name']:15s} | X:{lm['x']:.3f} Y:{lm['y']:.3f} Z:{lm['z']:.3f}"
            color = self._get_depth_color((lm['z'] + 1.0) / 2.0)
            cv2.putText(vis_frame, text, (10, y_text_start + idx * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_frame

# =====================================================================
# REAL-TIME RECOGNITION SYSTEM
# =====================================================================

class RealtimeSignLanguageSystem:
    """Complete real-time sign language recognition system"""
    
    def __init__(self, model_path=None, label_encoder=None, confidence_threshold=0.7):
        self.visualizer = AdvancedMediaPipeVisualizer(confidence_threshold=0.5)
        self.model = None
        self.label_encoder = label_encoder
        self.frame_buffer = deque(maxlen=16)  # 16 frames buffer
        self.landmark_buffer = deque(maxlen=16)
        self.prediction_confidence_threshold = confidence_threshold
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_history = deque(maxlen=10)
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model loaded: {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
    
    def predict_gesture(self, landmarks_seq, frames_seq):
        """Predict gesture from sequences"""
        if self.model is None or len(landmarks_seq) < 16:
            return None, 0
        
        try:
            # Prepare inputs
            landmarks_input = np.expand_dims(np.array(landmarks_seq), axis=0)
            frames_input = np.expand_dims(np.array(frames_seq), axis=0)
            
            # Predict
            probs = self.model.predict([landmarks_input, frames_input], verbose=0)
            pred_idx = np.argmax(probs[0])
            confidence = float(probs[0][pred_idx])
            
            if confidence >= self.prediction_confidence_threshold and self.label_encoder:
                gesture = self.label_encoder.inverse_transform([pred_idx])[0]
                return gesture, confidence
            
            return None, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0
    
    def run(self, video_source=0, display_landmarks=True, display_info=True):
        """Run real-time recognition system"""
        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*70)
        print("REAL-TIME SIGN LANGUAGE RECOGNITION WITH MEDIAPIPE")
        print("="*70)
        print("✓ 21-point hand landmark detection enabled")
        print("✓ Depth visualization enabled (Blue=Far, Green=Mid, Red=Close)")
        print("✓ Connection lines showing hand skeleton")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Toggle landmark labels")
        print("  'c' - Toggle connections")
        print("  'r' - Reset frame buffer")
        print("  'p' - Screenshot of landmarks")
        print("="*70 + "\n")
        
        frame_count = 0
        show_labels = display_landmarks
        show_connections = True
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Extract and visualize landmarks
            landmarks, annotated_frame, hand_info = self.visualizer.extract_and_visualize(
                frame,
                draw_labels=show_labels,
                draw_connections=show_connections
            )
            
            # Add frame and landmarks to buffers
            frame_resized = cv2.resize(annotated_frame, (224, 224)).astype('float32') / 255.0
            self.frame_buffer.append(frame_resized)
            
            if len(landmarks) > 0:
                self.landmark_buffer.append(landmarks[0])
            else:
                self.landmark_buffer.append(np.zeros(63))
            
            # Predict every 5 frames
            if frame_count % 5 == 0 and len(self.landmark_buffer) == 16:
                gesture, conf = self.predict_gesture(
                    list(self.landmark_buffer),
                    list(self.frame_buffer)
                )
                
                if gesture:
                    self.last_prediction = gesture
                    self.last_prediction_time = time.time()
                    self.prediction_history.append((gesture, conf))
            
            # Draw information
            if display_info:
                self.visualizer.draw_landmark_info(annotated_frame, hand_info, position=(10, 30))
            
            # Draw buffer status
            cv2.putText(annotated_frame, 
                       f"Buffer: {len(self.landmark_buffer)}/16",
                       (10, h - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw prediction
            if self.last_prediction:
                elapsed = time.time() - self.last_prediction_time
                if elapsed < 2:  # Show for 2 seconds
                    cv2.rectangle(annotated_frame, (20, h-120), (600, h-40), (0, 255, 0), -1)
                    cv2.putText(annotated_frame,
                               f"Gesture: {self.last_prediction}",
                               (40, h - 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
            
            # Display
            cv2.imshow("Real-Time Sign Language Recognition - Hand Landmarks", annotated_frame)
            
            # Create side panel with landmark details
            if hand_info['detected'] and display_landmarks:
                landmark_vis = self.visualizer.create_landmark_visualization(hand_info)
                cv2.imshow("21-Point Hand Landmarks Details", landmark_vis)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_labels = not show_labels
                print(f"Label display: {'ON' if show_labels else 'OFF'}")
            elif key == ord('c'):
                show_connections = not show_connections
                print(f"Connections: {'ON' if show_connections else 'OFF'}")
            elif key == ord('r'):
                self.frame_buffer.clear()
                self.landmark_buffer.clear()
                print("Buffer reset!")
            elif key == ord('p'):
                # Save screenshot
                if hand_info['detected']:
                    landmark_vis = self.visualizer.create_landmark_visualization(hand_info)
                    filename = f"landmark_screenshot_{screenshot_count:03d}.png"
                    cv2.imwrite(filename, landmark_vis)
                    cv2.imwrite(f"frame_{screenshot_count:03d}.png", annotated_frame)
                    print(f"✓ Saved: {filename}")
                    screenshot_count += 1
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ System shutdown complete")

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def create_demo_visualization():
    """Create a demo visualization of MediaPipe 21 landmarks"""
    
    fig = plt.figure(figsize=(14, 8))
    
    # Left panel: Hand skeleton
    ax1 = fig.add_subplot(121)
    
    # Synthetic landmark positions for demo
    np.random.seed(42)
    landmarks_x = np.random.rand(21) * 0.8 + 0.1
    landmarks_y = np.random.rand(21) * 0.8 + 0.1
    
    # Draw connections
    connections = AdvancedMediaPipeVisualizer.HAND_CONNECTIONS
    for start, end in connections:
        ax1.plot([landmarks_x[start], landmarks_x[end]],
                [landmarks_y[start], landmarks_y[end]],
                'g-', linewidth=2)
    
    # Draw landmarks with depth coloring
    z_coords = np.random.rand(21)
    scatter = ax1.scatter(landmarks_x, landmarks_y, s=200, c=z_coords,
                         cmap='cool', edgecolors='black', linewidth=2)
    
    # Add landmark labels
    landmark_names = AdvancedMediaPipeVisualizer.LANDMARK_NAMES
    for i, name in enumerate(landmark_names):
        ax1.annotate(f"{i}", (landmarks_x[i], landmarks_y[i]),
                    fontsize=8, ha='center', va='center')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(1, 0)
    ax1.set_aspect('equal')
    ax1.set_title('MediaPipe 21 Hand Landmarks\n(0=Wrist, 1-4=Thumb, 5-8=Index, etc.)',
                 fontweight='bold', fontsize=12)
    ax1.set_xlabel('X (normalized)')
    ax1.set_ylabel('Y (normalized)')
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Z (Depth)')
    
    # Right panel: Landmark info table
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    table_data = []
    for i in range(0, 21, 5):
        for j in range(i, min(i+5, 21)):
            name = landmark_names[j]
            table_data.append([f"{j}", name[:15]])
    
    table = ax2.table(cellText=table_data, colLabels=['ID', 'Landmark'],
                     cellLoc='left', loc='center',
                     colWidths=[0.15, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    ax2.set_title('21 Hand Landmark Definitions\n(MediaPipe Hands)',
                 fontweight='bold', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig('mediapipe_21_landmarks_guide.png', dpi=300, bbox_inches='tight')
    print("✓ Landmark guide saved: mediapipe_21_landmarks_guide.png")
    plt.show()

# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED REAL-TIME SIGN LANGUAGE RECOGNITION SYSTEM")
    print("With MediaPipe 21-Point Hand Landmark Detection & Visualization")
    print("="*70)
    
    # Option 1: Create demo visualization
    print("\n[1] Creating MediaPipe 21-Landmark Guide...")
    create_demo_visualization()
    
    # Option 2: Run real-time system
    print("\n[2] Initializing Real-Time System...")
    system = RealtimeSignLanguageSystem(confidence_threshold=0.7)
    
    # Uncomment to load a trained model
    # system.load_model('sign_language_recognition_model.h5')
    
    print("\n[3] Starting Real-Time Recognition...")
    system.run(video_source=0, display_landmarks=True, display_info=True)
    
    print("\n✓ Demo complete!")
