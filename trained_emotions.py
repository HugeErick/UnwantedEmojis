from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.preprocessor import to_categorical
import numpy as np
import cv2
from pathlib import Path

MODEL_PATH = "./models/emotion_model.hdf5"
FINETUNED_MODEL_PATH = "./models/emotion_model_finetuned.hdf5"
BATCH_SIZE = 32
EPOCHS = 10
IMAGE_SIZE = (48, 48)

# Emotion labels matching your folder structure
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_custom_dataset(dataset_path="./utils/custom_dataset"):
    """Load images from custom dataset folder structure"""
    images = []
    labels = []

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    print(f"Loading images from: {dataset_path}\n")

    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_folder = dataset_path / emotion

        if not emotion_folder.exists():
            print(
                f"Warning: Folder '{emotion}' not found, must have images in order to train properly"
            )
            break

        image_files = (
            list(emotion_folder.glob("*.jpg"))
            + list(emotion_folder.glob("*.jpeg"))
            + list(emotion_folder.glob("*.png"))
        )

        print(f"Loading {len(image_files)} images from '{emotion}' folder...")

        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE)
                    images.append(img)
                    labels.append(emotion_idx)
            except Exception as e:
                print(f"  Error loading {img_file.name}: {e}")

    if len(images) == 0:
        raise ValueError(
            "No images found! Please add images to the custom_dataset folders."
        )

    return np.array(images), np.array(labels)


def simple_augment(image):
    """Simple numpy-based augmentation"""
    img = image.copy()

    # Random horizontal flip
    if np.random.random() > 0.5:
        img = np.fliplr(img)

    # Random brightness adjustment
    if np.random.random() > 0.5:
        brightness_factor = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness_factor, 0, 1)

    # Add small noise
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)

    return img


def batch_generator(x, y, batch_size, augment=True):
    """Generate batches of data"""
    num_samples = len(x)
    indices = np.arange(num_samples)

    while True:
        np.random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_x = []
            for idx in batch_indices:
                img = x[idx]
                if augment:
                    img = simple_augment(img)
                batch_x.append(img)

            batch_x = np.array(batch_x)
            batch_y = y[batch_indices]

            yield batch_x, batch_y


def main():
    print("=" * 60)
    print("Custom Dataset Training")
    print("=" * 60 + "\n")

    # Load custom dataset
    try:
        print("Loading custom dataset...")
        faces, emotions = load_custom_dataset("./utils/custom_dataset")
        print(f"\nTotal images loaded: {len(faces)}")

        # Show distribution
        print("\nDataset distribution:")
        for idx, emotion in enumerate(EMOTIONS):
            count = np.sum(emotions == idx)
            print(f"  {emotion}: {count} images")

    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nPlease ensure:")
        print("1. Folder './utils/custom_dataset' exists")
        print(
            "2. It contains subfolders: angry, disgust, fear, happy, neutral, sad, surprise"
        )
        print("3. Each subfolder has .jpg, .jpeg, or .png images")
        return

    # Check minimum samples
    if len(faces) < 10:
        print("\nError: Not enough images! You need at least 10 images total.")
        print("Please add more images to the custom_dataset folders.")
        return

    # Split dataset
    print("\nSplitting dataset (80% train, 20% validation)...")
    num_samples = len(faces)

    # Shuffle indices
    indices = np.random.permutation(num_samples)
    split = int(num_samples * 0.8)

    train_idx = indices[:split]
    val_idx = indices[split:]

    x_train, y_train = faces[train_idx], emotions[train_idx]
    x_val, y_val = faces[val_idx], emotions[val_idx]

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    # Ensure data has channel dimension
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, -1)
        x_val = np.expand_dims(x_val, -1)

    # Convert labels to categorical
    num_classes = len(EMOTIONS)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    print(f"\nTraining samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Image shape: {x_train.shape[1:]}")
    print(f"Number of classes: {num_classes}")

    # Load model
    print("\nLoading existing model...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model exists at: {MODEL_PATH}")
        return

    # Setup callbacks
    checkpoint = ModelCheckpoint(
        FINETUNED_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, verbose=1, restore_best_weights=True
    )

    # Calculate steps per epoch
    steps_per_epoch = max(1, len(x_train) // BATCH_SIZE)
    validation_steps = max(1, len(x_val) // BATCH_SIZE)

    print("\nTraining configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation steps: {validation_steps}")

    # Create generators
    train_gen = batch_generator(x_train, y_train_cat, BATCH_SIZE, augment=True)
    val_gen = batch_generator(x_val, y_val_cat, BATCH_SIZE, augment=False)

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    try:
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=[checkpoint, early_stopping],
            verbose=1,
        )

        # Results
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"Best model saved to: {FINETUNED_MODEL_PATH}")
        print("\nFinal metrics:")
        print(f"  Training loss: {history.history['loss'][-1]:.4f}")
        print(f"  Validation loss: {history.history['val_loss'][-1]:.4f}")

        if "accuracy" in history.history:
            print(f"  Training accuracy: {history.history['accuracy'][-1]:.4f}")
            print(f"  Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
