import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/dev/null'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Disable JIT compilation entirely
os.environ['TF_DISABLE_JIT'] = '1'
from keras.models import Model
from keras.utils import Sequence
from keras.applications import EfficientNetB0
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from utils.preprocessor import to_categorical
import numpy as np
import cv2
from pathlib import Path

# Import dependencies for URL loading
try:
    import pandas as pd
    import requests
    from io import BytesIO
    from PIL import Image
    URL_SUPPORT = True
except ImportError as e:
    URL_SUPPORT = False
    MISSING_DEPS = str(e)

MODEL_PATH = "./src/models/emotion_model.hdf5"
FINETUNED_MODEL_PATH = "./src/models/emotion_model_finetuned.keras"
TARGET_IMAGES_PER_CLASS = 5000

# Emotion labels matching your folder structure
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def _pick_n(sequence, n=TARGET_IMAGES_PER_CLASS, rng=np.random.default_rng()):
    """Return *n* random items from *sequence* without replacement."""
    if len(sequence) <= n:
        return sequence[:]
    indices = rng.choice(len(sequence), size=n, replace=False)
    return [sequence[i] for i in indices]

def load_image_from_url(url, timeout=10):
    """Download and load image from URL"""
    if not URL_SUPPORT:
        return None
        
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        img_pil = Image.open(BytesIO(response.content))
        if img_pil.mode != 'L':
            img_pil = img_pil.convert('L')
        img = np.array(img_pil)
        return img
    except Exception as e:
        print(f"  Error loading URL: {e}")
        return None

def load_images_from_csv(csv_path, image_size):
    """Load images from URLs in a CSV file"""
    images = []
    
    if not URL_SUPPORT:
        print(f"  Warning: URL support not available. Missing dependencies: {MISSING_DEPS}")
        print("  Install with: pip install pandas pillow requests")
        return images
    
    try:
        df = pd.read_csv(csv_path)
        if 'url' not in df.columns:
            print(f"  Warning: 'url' column not found in {csv_path.name}")
            return images
        
        urls = df['url'].dropna().tolist()
        print(f"  Found {len(urls)} URLs in CSV")
        
        for idx, url in enumerate(urls, 1):
            if pd.isna(url) or not str(url).strip():
                continue
            img = load_image_from_url(str(url).strip())
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                if idx % 10 == 0:
                    print(f"    Loaded {idx}/{len(urls)} images...")
        
        print(f"  Successfully loaded {len(images)} images from URLs")
    except Exception as e:
        print(f"  Error reading CSV: {e}")
    
    return images

def load_custom_dataset(dataset_path="./src/utils/custom_dataset", image_size=(64, 64)):
    """Load images from custom dataset folder structure"""
    images, labels = [], []
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    print(f"Loading images from: {dataset_path}\n")

    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_folder = dataset_path / emotion

        if not emotion_folder.exists():
            print(f"Warning: Folder '{emotion}' not found, must have images in order to train properly")
            continue

        emotion_images = []
        
        # Method 1: Check for CSV file(s) with URLs
        csv_files = list(emotion_folder.glob("*.csv"))
        if csv_files:
            for csv_file in csv_files:
                print(f"Loading from CSV: '{emotion}/{csv_file.name}'")
                csv_images = load_images_from_csv(csv_file, image_size)
                emotion_images.extend(csv_images)
        
        # Method 2: Load local image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.avif", "*.bmp", "*.webp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(emotion_folder.glob(ext)))
            image_files.extend(list(emotion_folder.glob(ext.upper())))
        
        if image_files:
            print(f"Loading {len(image_files)} local images from '{emotion}' folder...")
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        emotion_images.append(img)
                except Exception as e:
                    print(f"  Error loading {img_file.name}: {e}")
        
        emotion_images = _pick_n(emotion_images, n=TARGET_IMAGES_PER_CLASS)
        images.extend(emotion_images)
        labels.extend([emotion_idx] * len(emotion_images))
        print(f"Total for '{emotion}': {len(emotion_images)} images\n")

    if len(images) == 0:
        raise ValueError("No images found! Please add images to the custom_dataset folders or create images.csv files with URLs.")

    return np.array(images), np.array(labels)

def convert_grayscale_to_rgb(images):
    """Convert grayscale images to RGB by replicating channels"""
    if len(images.shape) == 3:
        images = np.expand_dims(images, -1)
    rgb_images = np.repeat(images, 3, axis=-1)
    return rgb_images


class DataGenerator(Sequence):
    """Custom data generator compatible with Python 3.13"""
    def __init__(self, x, y, batch_size, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(x))
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.x) // self.batch_size
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        
        batch_x = self.x[batch_indices].copy()
        batch_y = self.y[batch_indices]
        
        # Normalize only - no augmentation
        batch_x = batch_x.astype('float32') / 255.0
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_transfer_learning_model(input_shape=(96, 96, 3), num_classes=7):
    """Create model using EfficientNetB0 with transfer learning"""
    print("\nCreating Transfer Learning Model (EfficientNetB0)...")
    
    # Ensure we're using the correct backend settings for RGB
    import keras.backend as K
    if hasattr(K, 'set_image_data_format'):
        K.set_image_data_format('channels_last')
    
    # Create base model
    try:
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(96, 96, 3),
            pooling=None
        )
        print(f"  Base model loaded: EfficientNetB0 ({len(base_model.layers)} layers)")
    except Exception as e:
        print(f"  Error loading pretrained weights: {e}")
        print("  Attempting to load without pretrained weights...")
        base_model = EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(96, 96, 3),
            pooling=None
        )
        print("  WARNING: Using EfficientNetB0 without ImageNet weights!")
        print("  Training may take longer and accuracy may be lower.")
    
    base_model.trainable = False
    print("  Base model frozen for Phase 1 training")
    
    inputs = Input(shape=(96, 96, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='bn_1')(x)
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dense(128, activation='relu', name='dense_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    print(f"  Model created with {model.count_params():,} parameters\n")
    
    return model, base_model

def train_transfer_learning_model():
    """Train new transfer learning model (Python 3.13 compatible)"""
    BATCH_SIZE = 16
    IMAGE_SIZE = (96, 96)
    
    print("=" * 70)
    print("GPU-ACCELERATED TRANSFER LEARNING")
    print("=" * 70)
    print("Training with custom data generator (Python 3.13 compatible)")
    print("No augmentation - using pre-augmented dataset")
    print("Expected accuracy: 80-85%")
    print("=" * 70 + "\n")

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Using GPU: {gpus[0].name}")
            print("  GPU Memory: ~3.5GB available\n")
        else:
            print("⚠️  No GPU detected, using CPU\n")
    except ImportError:
        print("❌ TensorFlow not installed\n")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}\n")

    # Load dataset
    try:
        print("Loading custom dataset...")
        faces, emotions = load_custom_dataset(image_size=IMAGE_SIZE)
        print(f"\nTotal images loaded: {len(faces)}")

        print("\nDataset distribution:")
        for idx, emotion in enumerate(EMOTIONS):
            count = np.sum(emotions == idx)
            print(f"  {emotion}: {count} images")

    except Exception as e:
        print(f"\nError loading dataset: {e}")
        return

    if len(faces) < 10:
        print("\nError: Not enough images!")
        return

    # Split dataset
    print("\nSplitting dataset (80% train, 20% validation)...")
    indices = np.random.permutation(len(faces))
    split = int(len(faces) * 0.8)
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    x_train, y_train = faces[train_idx], emotions[train_idx]
    x_val, y_val = faces[val_idx], emotions[val_idx]

    # Convert to RGB
    print("\nConverting grayscale to RGB...")
    x_train = convert_grayscale_to_rgb(x_train)
    x_val = convert_grayscale_to_rgb(x_val)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Image shape: {x_train.shape[1:]}")

    # Convert labels
    num_classes = len(EMOTIONS)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    # Create generators (Python 3.13 compatible) - NO AUGMENTATION
    train_gen = DataGenerator(x_train, y_train_cat, BATCH_SIZE, shuffle=True)
    val_gen = DataGenerator(x_val, y_val_cat, BATCH_SIZE, shuffle=False)

    # Create model
    model, base_model = create_transfer_learning_model(
        input_shape=(*IMAGE_SIZE, 3),
        num_classes=num_classes
    )

    # PHASE 1: Train head with frozen base
    print("=" * 70)
    print("PHASE 1: Training Custom Head (Base Frozen)")
    print("=" * 70)
    print("Training only the new layers, EfficientNet stays frozen")
    print("Expected: ~75-78% accuracy after this phase\n")
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    checkpoint1 = ModelCheckpoint(
        './src/models/emotion_transfer_phase1.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping1 = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr1 = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    print("Starting Phase 1 training...\n")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        callbacks=[checkpoint1, early_stopping1, reduce_lr1],
        verbose=1
    )
    
    best_val_acc_p1 = max(history1.history['val_accuracy'])
    print(f"\n✓ Phase 1 Complete! Best validation accuracy: {best_val_acc_p1:.4f} ({best_val_acc_p1*100:.2f}%)\n")

    # PHASE 2: Fine-tune base model
    print("=" * 70)
    print("PHASE 2: Fine-Tuning Base Model")
    print("=" * 70)
    print("Unfreezing top 30% of EfficientNet for fine-tuning")
    print("Expected: ~80-85% accuracy after this phase\n")
    
    # Unfreeze top 30%
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * 0.7)
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Unfrozen {trainable_layers}/{total_layers} base model layers\n")
    
    # Recompile with lower LR
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    checkpoint2 = ModelCheckpoint(
        FINETUNED_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping2 = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr2 = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
    
    print("Starting Phase 2 training...\n")
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        callbacks=[checkpoint2, early_stopping2, reduce_lr2],
        verbose=1
    )
    
    best_val_acc_p2 = max(history2.history['val_accuracy'])
    print(f"\n✓ Phase 2 Complete! Best validation accuracy: {best_val_acc_p2:.4f} ({best_val_acc_p2*100:.2f}%)")
    
    # Final results
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {FINETUNED_MODEL_PATH}")
    print(f"\nPhase 1 Best Accuracy: {best_val_acc_p1:.4f} ({best_val_acc_p1*100:.2f}%)")
    print(f"Phase 2 Best Accuracy: {best_val_acc_p2:.4f} ({best_val_acc_p2*100:.2f}%)")
    print(f"Total Improvement: +{(best_val_acc_p2 - 0.70) * 100:.2f} percentage points from baseline")
    print("\nThis model uses RGB images at 96x96 resolution.")
    print("Make sure your inference code handles RGB conversion!")

def main():
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING - EMOTION DETECTION TRAINING")
    print("=" * 70)
    print("\nUsing EfficientNetB0 pre-trained on ImageNet")
    print("GPU-accelerated training (if available)")
    print("No augmentation - dataset already augmented")
    print("Expected accuracy: 80-85%")
    print("=" * 70 + "\n")
    
    train_transfer_learning_model()

if __name__ == "__main__":
    main()