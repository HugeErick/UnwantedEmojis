import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import cv2
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical, Sequence
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import json
from datetime import datetime

# Configuration
TARGET_IMAGES_PER_CLASS = 5200
IMAGE_SIZE = (128, 128)  # This is the image size 
BATCH_SIZE = 16
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def _pick_n(sequence, n=TARGET_IMAGES_PER_CLASS, rng=np.random.default_rng()):
    """Return n random items from sequence without replacement."""
    if len(sequence) <= n:
        return sequence[:]
    indices = rng.choice(len(sequence), size=n, replace=False)
    return [sequence[i] for i in indices]

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
            print(f"Warning: Folder '{emotion}' not found")
            continue

        # Load local image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.avif", "*.bmp", "*.webp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(emotion_folder.glob(ext)))
            image_files.extend(list(emotion_folder.glob(ext.upper())))
        
        if image_files:
            print(f"Loading {len(image_files)} images from '{emotion}' folder...")
            emotion_images = []
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        emotion_images.append(img)
                except Exception as e:
                    pass
            
            emotion_images = _pick_n(emotion_images, n=TARGET_IMAGES_PER_CLASS)
            images.extend(emotion_images)
            labels.extend([emotion_idx] * len(emotion_images))
            print(f"Total for '{emotion}': {len(emotion_images)} images\n")

    if len(images) == 0:
        raise ValueError("No images found!")

    return np.array(images), np.array(labels)


class AugmentedDataGenerator(Sequence):
    """Data generator with strong augmentation"""
    def __init__(self, x, y, batch_size, shuffle=True, augment=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
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
        
        if self.augment:
            batch_x = self._augment_batch(batch_x)
        
        # Normalize to [0, 1]
        batch_x = batch_x.astype('float32') / 255.0
        
        # Add channel dimension
        if len(batch_x.shape) == 3:
            batch_x = np.expand_dims(batch_x, -1)
        
        return batch_x, batch_y
    
    def _augment_batch(self, batch):
        """Apply aggressive augmentation"""
        augmented = []
        for img in batch:
            # Random horizontal flip
            if random.random() > 0.5:
                img = np.fliplr(img)
            
            # Random brightness (±30%)
            brightness = random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 255)
            
            # Random contrast
            alpha = random.uniform(0.8, 1.2)
            img = np.clip(128 + alpha * (img - 128), 0, 255)
            
            # Random rotation (-20 to +20 degrees)
            angle = random.uniform(-20, 20)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=128)
            
            # Random shift
            shift_x = random.randint(-5, 5)
            shift_y = random.randint(-5, 5)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (w, h), borderValue=128)
            
            # Random zoom
            if random.random() > 0.7:
                zoom = random.uniform(0.9, 1.1)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), 0, zoom)
                img = cv2.warpAffine(img, M, (w, h), borderValue=128)
            
            augmented.append(img)
        
        return np.array(augmented)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_custom_cnn(input_shape=(64, 64, 1), num_classes=7):
    """
    Create a custom CNN optimized for emotion detection.
    """
    print("\nCreating Custom CNN Model...")
    
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 3
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Classifier
    x = Flatten()(x)
    x = Dense(512, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    print(f"Model created with {model.count_params():,} parameters")
    
    return model


def plot_training_history(history, save_path='./training_results'):
    """Plot training and validation metrics"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_history.png', dpi=300, bbox_inches='tight')
    print(f"Saved training history plot to {save_path}/training_history.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='./training_results'):
    """Plot confusion matrix"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax)
    
    ax.set_title('Confusion Matrix (%)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    
    # Add counts in cells
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            text = ax.text(j + 0.5, i + 0.7, f'n={cm[i, j]}',
                          ha="center", va="center", color="gray", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}/confusion_matrix.png")
    plt.close()
    
    return cm


def plot_per_class_metrics(y_true, y_pred, save_path='./training_results'):
    """Plot per-class precision, recall, and F1-score"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(EMOTIONS))
    )
    
    x = np.arange(len(EMOTIONS))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/per_class_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved per-class metrics to {save_path}/per_class_metrics.png")
    plt.close()
    
    return precision, recall, f1, support


def save_metrics_report(history, y_true, y_pred, cm, precision, recall, f1, support, 
                       training_time, save_path='./training_results'):
    """Save comprehensive metrics report"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Calculate overall metrics
    accuracy = np.mean(y_true == y_pred)
    
    # Create detailed report
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_configuration': {
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE,
            'target_images_per_class': TARGET_IMAGES_PER_CLASS,
            'total_epochs': len(history.history['accuracy']),
            'training_time_seconds': training_time
        },
        'overall_metrics': {
            'accuracy': float(accuracy),
            'best_train_accuracy': float(max(history.history['accuracy'])),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        },
        'per_class_metrics': {}
    }
    
    # Add per-class metrics
    for idx, emotion in enumerate(EMOTIONS):
        report['per_class_metrics'][emotion] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1_score': float(f1[idx]),
            'support': int(support[idx]),
            'samples_percentage': float(support[idx] / len(y_true) * 100)
        }
    
    # Save as JSON
    with open(f'{save_path}/metrics_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Saved metrics report to {save_path}/metrics_report.json")
    
    # Save as readable text
    with open(f'{save_path}/metrics_report.txt', 'w') as f:
        f.write("="*20 + "\n")
        f.write("EMOTION DETECTION MODEL - TRAINING REPORT\n")
        f.write("="*20 + "\n\n")
        
        f.write(f"Generated: {report['timestamp']}\n")
        f.write(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*20 + "\n")
        f.write(f"Image Size: {IMAGE_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Total Epochs Trained: {len(history.history['accuracy'])}\n")
        f.write(f"Target Images per Class: {TARGET_IMAGES_PER_CLASS}\n\n")
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*20 + "\n")
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Best Training Accuracy: {report['overall_metrics']['best_train_accuracy']*100:.2f}%\n")
        f.write(f"Best Validation Accuracy: {report['overall_metrics']['best_val_accuracy']*100:.2f}%\n")
        f.write(f"Final Training Loss: {report['overall_metrics']['final_train_loss']:.4f}\n")
        f.write(f"Final Validation Loss: {report['overall_metrics']['final_val_loss']:.4f}\n\n")
        
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-"*20 + "\n")
        f.write(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*20 + "\n")
        
        for emotion in EMOTIONS:
            metrics = report['per_class_metrics'][emotion]
            f.write(f"{emotion:<12} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} "
                   f"{metrics['f1_score']:<12.3f} {metrics['support']:<10}\n")
        
        f.write("\n" + "="*20 + "\n")
        f.write("CONFUSION MATRIX (Raw Counts)\n")
        f.write("="*20 + "\n\n")
        
        # Print confusion matrix
        f.write(f"{'':>12}")
        for emotion in EMOTIONS:
            f.write(f"{emotion[:8]:>10}")
        f.write("\n")
        
        for i, emotion in enumerate(EMOTIONS):
            f.write(f"{emotion:<12}")
            for j in range(len(EMOTIONS)):
                f.write(f"{cm[i][j]:>10}")
            f.write("\n")
    
    print(f"Saved readable report to {save_path}/metrics_report.txt")


def train_model():
    """Train the emotion detection model with comprehensive metrics"""
    import time
    start_time = time.time()
    
    print("\n" + "="*70)
    print("CUSTOM CNN EMOTION DETECTION TRAINING")
    print("="*20)
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Architecture: Custom CNN (proven for emotion detection)")
    print("="*20 + "\n")
    
    # Load dataset
    try:
        print("Loading dataset...")
        faces, emotions = load_custom_dataset(image_size=IMAGE_SIZE)
        print(f"\nTotal images loaded: {len(faces)}")
        
        print("\nDataset distribution:")
        for idx, emotion in enumerate(EMOTIONS):
            count = np.sum(emotions == idx)
            print(f"  {emotion}: {count} images")
    
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        return
    
    if len(faces) < 100:
        print("\nError: Not enough images!")
        return
    
    # Split dataset (80/20)
    print("\nSplitting dataset (80% train, 20% validation)...")
    indices = np.random.permutation(len(faces))
    split = int(len(faces) * 0.8)
    
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    x_train, y_train = faces[train_idx], emotions[train_idx]
    x_val, y_val = faces[val_idx], emotions[val_idx]
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    
    # Convert labels to categorical
    num_classes = len(EMOTIONS)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    
    # Create data generators
    print("\nCreating data generators with augmentation...")
    train_gen = AugmentedDataGenerator(
        x_train, y_train_cat, 
        BATCH_SIZE, 
        shuffle=True, 
        augment=True
    )
    
    val_gen = AugmentedDataGenerator(
        x_val, y_val_cat,
        BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    # Create model
    model = create_custom_cnn(
        input_shape=(*IMAGE_SIZE, 1),
        num_classes=num_classes
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        './src/models/emotion_model_finetuned.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train
    print("\n" + "="*20)
    print("STARTING TRAINING")
    print("="*20)
    print("Expected: 50-70% accuracy after 20-40 epochs")
    print("="*20 + "\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on validation set
    print("\n" + "="*20)
    print("GENERATING PREDICTIONS AND METRICS")
    print("="*20 + "\n")
    
    # Get predictions
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_val[:len(y_pred)]  # Ensure same length
    
    # Generate all plots and metrics
    print("\nGenerating visualizations...")
    plot_training_history(history)
    cm = plot_confusion_matrix(y_true, y_pred)
    precision, recall, f1, support = plot_per_class_metrics(y_true, y_pred)
    
    # Save comprehensive report
    save_metrics_report(history, y_true, y_pred, cm, precision, recall, f1, 
                       support, training_time)
    
    # Print summary
    best_val_acc = max(history.history['val_accuracy'])
    best_train_acc = max(history.history['accuracy'])
    test_accuracy = np.mean(y_true == y_pred)
    
    print("\n" + "="*20)
    print("TRAINING COMPLETE!")
    print("="*20)
    print(f"Training Time: {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"Best Training Accuracy: {best_train_acc:.4f} ({best_train_acc*100:.2f}%)")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"\nModel saved to: ./src/models/emotion_model_finetuned.keras")
    print(f"Results saved to: ./training_results/")
    print("="*20)
    
    return model, history


if __name__ == "__main__":
    train_model()
