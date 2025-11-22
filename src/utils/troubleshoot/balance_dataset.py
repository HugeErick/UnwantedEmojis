"""
Dataset Balancing Script
========================
This script:
1. Extracts Disgust.jpg images from the downloads dataset
2. Augments disgust images with variations to reach 3000
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

# Configuration
DOWNLOADS_DATASET = Path.home() / "downloads" / "i"
TARGET_DATASET = Path.home() / "devNest" / "UnwantedEmojis" / "src" / "utils" / "custom_dataset"
TARGET_IMAGES_PER_CLASS = 5000
IMAGE_SIZE = (64, 64)

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Mapping from dataset filenames to emotion folders
FILENAME_TO_EMOTION = {
    "Anger.jpg": "angry",
    "Disgust.jpg": "disgust",
    "Fear.jpg": "fear",
    "Happy.jpg": "happy",
    "Neutral.jpg": "neutral",
    "Sad.jpg": "sad",
    "Surprised.jpg": "surprise"
}


def augment_image(image, seed=None):
    """
    Apply random augmentations to an image
    Returns a slightly modified version
    """
    if seed is not None:
        np.random.seed(seed)
    
    img = image.copy()
    
    # Random horizontal flip (50% chance)
    if np.random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # Random rotation (-15 to +15 degrees)
    angle = np.random.uniform(-15, 15)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random brightness adjustment (0.7 to 1.3)
    brightness_factor = np.random.uniform(0.7, 1.3)
    img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
    
    # Random contrast adjustment
    contrast_factor = np.random.uniform(0.8, 1.2)
    mean = img.mean()
    img = np.clip((img - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
    
    # Random noise (small amount)
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 3, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Random translation (-5 to +5 pixels)
    tx = np.random.randint(-5, 6)
    ty = np.random.randint(-5, 6)
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return img


def extract_disgust_images():
    """Extract only Disgust.jpg images from downloads dataset"""
    print("=" * 60)
    print("STEP 1: Extracting Disgust Images")
    print("=" * 60)
    
    if not DOWNLOADS_DATASET.exists():
        print(f"Error: Source dataset not found at {DOWNLOADS_DATASET}")
        return 0
    
    disgust_folder = TARGET_DATASET / "disgust"
    disgust_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all numbered folders (0-18)
    folders = sorted([f for f in DOWNLOADS_DATASET.iterdir() if f.is_dir() and f.name.isdigit()])
    
    extracted_count = 0
    
    print(f"\nFound {len(folders)} folders to process")
    print(f"Target: {disgust_folder}\n")
    
    for folder in tqdm(folders, desc="Extracting disgust images"):
        disgust_file = folder / "Disgust.jpg"
        
        if disgust_file.exists():
            # Create unique filename
            target_file = disgust_folder / f"disgust_extracted_{folder.name}.jpg"
            
            # Copy the file
            shutil.copy2(disgust_file, target_file)
            extracted_count += 1
    
    print(f"\nExtracted {extracted_count} disgust images")
    return extracted_count


def augment_disgust_to_target():
    """Augment disgust images to reach TARGET_IMAGES_PER_CLASS"""
    print("\n" + "=" * 60)
    print("STEP 2: Augmenting Disgust Images")
    print("=" * 60)
    
    disgust_folder = TARGET_DATASET / "disgust"
    
    # Get all existing disgust images (including from CSV)
    existing_images = list(disgust_folder.glob("*.jpg")) + \
                     list(disgust_folder.glob("*.png")) + \
                     list(disgust_folder.glob("*.jpeg"))
    
    current_count = len(existing_images)
    print(f"\nCurrent disgust images: {current_count}")
    print(f"Target: {TARGET_IMAGES_PER_CLASS}")
    
    if current_count >= TARGET_IMAGES_PER_CLASS:
        print("Already have enough images!")
        return
    
    needed = TARGET_IMAGES_PER_CLASS - current_count
    print(f"Need to create: {needed} augmented images\n")
    
    # Load all existing images
    base_images = []
    for img_path in tqdm(existing_images, desc="Loading base images"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            base_images.append(img)
    
    if not base_images:
        print("Error: No valid base images found!")
        return
    
    print(f"Loaded {len(base_images)} valid base images")
    print(f"Creating {needed} augmented versions...\n")
    
    # Create augmented images
    created = 0
    for i in tqdm(range(needed), desc="Creating augmented images"):
        # Pick a random base image
        base_img = base_images[i % len(base_images)]
        
        # Augment it with unique seed
        augmented = augment_image(base_img, seed=i)
        
        # Save with unique name
        output_path = disgust_folder / f"disgust_augmented_{current_count + i:05d}.jpg"
        cv2.imwrite(str(output_path), augmented)
        created += 1
    
    print(f"\nCreated {created} augmented images")
    print(f"Total disgust images: {current_count + created}")


def verify_all_classes():
    """Check current status of all emotion classes (no modifications)"""
    print("\n" + "=" * 60)
    print("STEP 3: Checking All Classes Status")
    print("=" * 60)
    
    for emotion in EMOTIONS:
        emotion_folder = TARGET_DATASET / emotion
        
        if not emotion_folder.exists():
            print(f"\n{emotion.upper()}:")
            print("Folder not found")
            continue
        
        # Get all images
        all_images = list(emotion_folder.glob("*.jpg")) + \
                    list(emotion_folder.glob("*.png")) + \
                    list(emotion_folder.glob("*.jpeg"))
        
        current_count = len(all_images)
        
        print(f"\n{emotion.upper()}:")
        print(f"  Current: {current_count} images")
        
        if current_count >= TARGET_IMAGES_PER_CLASS:
            print(f"Has enough images (will use {TARGET_IMAGES_PER_CLASS} during training)")
        else:
            print(f"Only {current_count} images (need {TARGET_IMAGES_PER_CLASS})")


def limit_class_to_target(emotion: str) -> int:
    """
    Ensure that *emotion* folder contains exactly TARGET_IMAGES_PER_CLASS
    samples by copying originals from the downloads dataset.
    Returns the number of images that had to be copied.
    """
    emotion_folder = TARGET_DATASET / emotion
    emotion_folder.mkdir(parents=True, exist_ok=True)

    # how many do we already have?
    current = list(emotion_folder.glob("*.jpg")) + \
              list(emotion_folder.glob("*.png")) + \
              list(emotion_folder.glob("*.jpeg"))
    current_count = len(current)

    if current_count >= TARGET_IMAGES_PER_CLASS:
        return 0          # nothing to do

    needed = TARGET_IMAGES_PER_CLASS - current_count
    copied = 0

    # map emotion -> filename  (reverse of FILENAME_TO_EMOTION)
    emotion_to_filename = {v: k for k, v in FILENAME_TO_EMOTION.items()}
    source_filename = emotion_to_filename.get(emotion)
    if source_filename is None:
        print(f"No mapping for {emotion} – skipping class")
        return 0

    # collect every occurrence of that file in the numbered sub-folders
    source_paths = []
    for numbered_dir in sorted(DOWNLOADS_DATASET.iterdir()):
        if numbered_dir.is_dir() and numbered_dir.name.isdigit():
            candidate = numbered_dir / source_filename
            if candidate.exists():
                source_paths.append(candidate)

    if not source_paths:
        print(f"️No source images found for {emotion}")
        return 0

    # copy until we reach the target
    for i in range(needed):
        src = source_paths[i % len(source_paths)]
        dst = emotion_folder / f"{emotion}_balanced_{current_count + i:05d}.jpg"
        shutil.copy2(src, dst)
        copied += 1

    return copied


def limit_all_classes_to_target():
    """Bring only the needed emotion class up (or down) to TARGET_IMAGES_PER_CLASS."""
    print("\n" + "=" * 60)
    print("STEP 3: Balancing all emotion classes")
    print("=" * 60)

    for emotion in EMOTIONS:
        if emotion == "disgust":
            continue          # already handled in step 2
        copied = limit_class_to_target(emotion)
        print(f"{emotion:10s}: copied {copied} images")

def verify_balance():
    """Verify that all can be balanced"""
    print("\n" + "=" * 60)
    print("VERIFICATION: Final Dataset Balance")
    print("=" * 60 + "\n")
    
    total_images = 0
    all_balanced = True
    
    for emotion in EMOTIONS:
        emotion_folder = TARGET_DATASET / emotion
        
        if not emotion_folder.exists():
            print(f"{emotion:10s}:FOLDER NOT FOUND")
            all_balanced = False
            continue
        
        images = list(emotion_folder.glob("*.jpg")) + \
                list(emotion_folder.glob("*.png")) + \
                list(emotion_folder.glob("*.jpeg"))
        
        count = len(images)
        total_images += count
        
        status = "ok" if count >= TARGET_IMAGES_PER_CLASS else "unsufficient"
        print(f"{emotion:10s}: {count:5d} images {status}")
        
        if status == "unsufficient":
            all_balanced = False
        else:
            all_balanced = True
    
    print(f"\n{'Total':10s}: {total_images:5d} images")
    print(f"Target total: {TARGET_IMAGES_PER_CLASS * len(EMOTIONS)} images")
    
    if all_balanced:
        print("\nSUCCESS! All classes are perfectly balanced!")
    else:
        print("\nWARNING: Dataset is not balanced!")
    
    return all_balanced

def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("EMOTION DATASET BALANCER  –  DISGUST-ONLY TOP-UP")
    print("=" * 60)
    print(f"\nTarget images for disgust : {TARGET_IMAGES_PER_CLASS}")
    print(f"Source dataset            : {DOWNLOADS_DATASET}")
    print(f"Disgust destination       : {TARGET_DATASET / 'disgust'}")
    input("\nPress Enter to start …")

    try:
        # 1. copy every Disgust.jpg we can find
        extracted = extract_disgust_images()
        if extracted == 0:
            print("\n❌ No Disgust.jpg files found – aborting.")
            return

        # 2. augment until we have 3000
        augment_disgust_to_target()

        # 3. quick sanity check
        verify_balance()

        print("\n" + "=" * 60)
        print("✅ DONE – disgust folder now contains 3000 images.")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()