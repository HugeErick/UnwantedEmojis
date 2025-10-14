import os
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from statistics import mode
from pathlib import Path
import logging

# Fix GPU memory issues and cuDNN problems
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow logging

# force the CPU (its ok)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logger = logging.getLogger(__name__)
        logger.info(f"GPU memory growth enabled for {len(physical_devices)} device(s)")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(f"GPU configuration warning: {e}")
else:
    print("No GPU devices found, using CPU")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modern imports - we'll create these utility functions inline
class ModernEmotionDetector:
    def __init__(self):
        self.USE_WEBCAM = True
        
        # Modern emotion labels (FER2013 dataset)
        self.emotion_labels = {
            0: 'angry',
            1: 'disgust', 
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }
        
        # Parameters
        self.emotion_model_path = './models/emotion_model.hdf5'
        self.frame_window = 10
        self.emotion_offsets = (20, 40)
        self.emotion_window = []
        
        # Modern color mapping
        self.color_map = {
            'angry': (255, 0, 0),      # Red
            'disgust': (0, 128, 0),    # Dark Green  
            'fear': (128, 0, 128),     # Purple
            'happy': (255, 255, 0),    # Yellow
            'sad': (0, 0, 255),        # Blue
            'surprise': (0, 255, 255), # Cyan
            'neutral': (0, 255, 0)     # Green
        }
        
        self.load_models()
    
    def load_models(self):
        """Load face detection and emotion classification models"""
        try:
            # Load face cascade
            cascade_path = './models/haarcascade_frontalface_default.xml'
            if not Path(cascade_path).exists():
                # Try OpenCV's built-in cascade
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            else:
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Load emotion model with modern TensorFlow
            logger.info("Loading emotion classification model...")
            if Path(self.emotion_model_path).exists():
                # Try loading the existing model
                try:
                    self.emotion_classifier = keras.models.load_model(
                        self.emotion_model_path, 
                        compile=False
                    )
                    logger.info("Legacy model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load legacy model: {e}")
                    self.create_modern_model()
            else:
                logger.warning("No existing model found, creating modern model")
                self.create_modern_model()
            
            # Get input shape
            self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
            logger.info(f"Model input shape: {self.emotion_target_size}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def create_modern_model(self):
        """Create a modern CNN model for emotion detection"""
        logger.info("Creating modern emotion classification model...")
        
        model = keras.Sequential([
            keras.layers.Input(shape=(48, 48, 1)),
            
            # First Conv Block
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Second Conv Block
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Third Conv Block
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Dense layers
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(7, activation='softmax')  # 7 emotions
        ])
        
        # Compile with modern optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.emotion_classifier = model
        logger.info("Modern model created successfully")
    
    def preprocess_input(self, x, v2=True):
        """Modern preprocessing function"""
        x = x.astype('float32')
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x
    
    def apply_offsets(self, face_coordinates, offsets):
        """Apply offsets to face bounding box"""
        x, y, width, height = face_coordinates
        x_offset, y_offset = offsets
        
        x1 = max(0, x - x_offset)
        y1 = max(0, y - y_offset)
        x2 = x + width + x_offset
        y2 = y + height + y_offset
        
        return x1, x2, y1, y2
    
    def draw_bounding_box(self, face_coordinates, image, color):
        """Draw bounding box around face"""
        x, y, w, h = face_coordinates
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    def draw_text(self, face_coordinates, image, text, color, x_offset=0, y_offset=0, 
                  font_scale=1, thickness=2):
        """Draw text on image"""
        x, y, w, h = face_coordinates
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size for better positioning
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + x_offset
        text_y = y + y_offset
        
        # Draw background rectangle for better text visibility
        cv2.rectangle(image, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)
    
    def detect_emotions(self, frame):
        """Detect emotions in a frame"""
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for face_coordinates in faces:
            x1, x2, y1, y2 = self.apply_offsets(face_coordinates, self.emotion_offsets)
            
            # Extract and preprocess face
            gray_face = gray_image[y1:y2, x1:x2]
            
            if gray_face.size == 0:
                continue
                
            try:
                gray_face = cv2.resize(gray_face, self.emotion_target_size)
            except Exception as e:
                logger.warning(f"Failed to resize face: {e}")
                continue
            
            # Preprocess
            gray_face = self.preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, axis=0)
            gray_face = np.expand_dims(gray_face, axis=-1)
            
            # Predict emotion with error handling
            try:
                emotion_prediction = self.emotion_classifier.predict(gray_face, verbose=0)
            except Exception as prediction_error:
                logger.warning(f"GPU prediction failed, trying CPU: {prediction_error}")
                # Force CPU prediction as fallback
                with tf.device('/CPU:0'):
                    try:
                        emotion_prediction = self.emotion_classifier.predict(gray_face, verbose=0)
                        logger.info("CPU prediction successful")
                    except Exception as cpu_error:
                        logger.error(f"Both GPU and CPU prediction failed: {cpu_error}")
                        continue
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = self.emotion_labels[emotion_label_arg]
            
            # Update emotion window for smoothing
            self.emotion_window.append(emotion_text)
            if len(self.emotion_window) > self.frame_window:
                self.emotion_window.pop(0)
            
            # Get mode emotion
            try:
                emotion_mode = mode(self.emotion_window)
            except Exception as e:
                logger.debug(f"Mode calculation failed: {e}")
                emotion_mode = emotion_text
            
            # Get color
            base_color = self.color_map.get(emotion_text, (0, 255, 0))
            color = (emotion_probability * np.array(base_color)).astype(int).tolist()
            
            # Draw bounding box and text
            self.draw_bounding_box(face_coordinates, rgb_image, color)
            self.draw_text(
                face_coordinates, rgb_image, 
                f"{emotion_mode} ({emotion_probability:.2f})",
                color, 0, -10, 0.7, 2
            )
        
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    def find_camera(self):
        """Find available camera devices"""
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                logger.info(f"Found camera at index {i}")
                cap.release()
                return i
        return None
    
    def run(self):
        """Main execution loop"""
        # Initialize video capture with better camera detection
        cap = None
        
        if self.USE_WEBCAM:
            logger.info("Searching for available cameras...")
            camera_index = self.find_camera()
            
            if camera_index is not None:
                cap = cv2.VideoCapture(camera_index)
                logger.info(f"Using camera at index {camera_index}")
            else:
                # Try different backends
                backends_to_try = [
                    cv2.CAP_V4L2,    # Linux
                    cv2.CAP_DSHOW,   # Windows
                    cv2.CAP_AVFOUNDATION,  # macOS
                    cv2.CAP_ANY      # Any available
                ]
                
                for backend in backends_to_try:
                    try:
                        cap = cv2.VideoCapture(0, backend)
                        if cap.isOpened():
                            logger.info(f"Using camera with backend {backend}")
                            break
                        cap.release()
                    except Exception as e:
                        logger.debug(f"Backend {backend} failed: {e}")
                        continue
                
                if cap is None or not cap.isOpened():
                    print("No camera detected. Please run the troubleshoot scripts in utils/troubleshoot/.")
                    return
        
        if not self.USE_WEBCAM:
            demo_path = './demo/dinner.mp4'
            if Path(demo_path).exists():
                cap = cv2.VideoCapture(demo_path)
                logger.info(f"Using demo video: {demo_path}")
            else:
                logger.error("No demo video found. Creating test pattern.")
                # Create a simple test pattern instead
                return self.run_test_pattern()
        
        if cap is None or not cap.isOpened():
            print("No camera detected. Please run the troubleshoot scripts in utils/troubleshoot/.")
            return
        
        # Set camera properties for better performance
        if self.USE_WEBCAM:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        cv2.namedWindow('Emotion Detection', cv2.WINDOW_AUTOSIZE)
        logger.info("Starting emotion detection... Press 'q' to quit")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Process frame
                processed_frame = self.detect_emotions(frame)
                
                # Display
                cv2.imshow('Emotion Detection', processed_frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during execution: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Emotion detection stopped")
    
    def run_test_pattern(self):
        """Run with a test pattern when no camera/video is available"""
        logger.info("Running with test pattern (no camera/video available)")
        
        # Create a simple test image with a face-like pattern
        height, width = 480, 640
        test_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a simple face pattern
        center = (width // 2, height // 2)
        # Face outline
        cv2.circle(test_frame, center, 100, (255, 255, 255), 2)
        # Eyes
        cv2.circle(test_frame, (center[0] - 30, center[1] - 30), 10, (255, 255, 255), -1)
        cv2.circle(test_frame, (center[0] + 30, center[1] - 30), 10, (255, 255, 255), -1)
        # Mouth
        cv2.ellipse(test_frame, (center[0], center[1] + 30), (30, 15), 0, 0, 180, (255, 255, 255), 2)
        
        cv2.namedWindow('Emotion Detection - Test Pattern', cv2.WINDOW_AUTOSIZE)
        
        # Add instruction text
        cv2.putText(test_frame, "No camera/video found - Test Pattern", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(test_frame, "Press 'q' to quit", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        logger.info("Test pattern ready. Press 'q' to quit")
        
        try:
            while True:
                cv2.imshow('Emotion Detection - Test Pattern', test_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cv2.destroyAllWindows()
            logger.info("Test pattern stopped")
