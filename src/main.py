import sys
from pathlib import Path
from emotions import ModernEmotionDetector

def check_finetuned_model_exists():
    """Check if a fine-tuned model exists"""
    keras_path = Path('./src/models/emotion_model_finetuned.keras')
    hdf5_path = Path('./src/models/emotion_model_finetuned.hdf5')
    return keras_path.exists() or hdf5_path.exists()

def main():
    while True:
        print("\n" + "="*20)
        print("       EMOTION DETECTION SYSTEM")
        print("="*20)
        print("\n==== Main Menu ====")
        print("1. Run Emotion Detection (Original Model)")
        print("2. Run Emotion Detection (Fine-tuned Model)")
        print("3. Train/Fine-tune Model")
        print("0. Exit")
        print("\n" + "-"*20)
        
        # Check if fine-tuned model exists
        has_finetuned = check_finetuned_model_exists()
        if has_finetuned:
            print("Fine-tuned model detected")
        else:
            print("No fine-tuned model found (use option 3 to train)")
        print("-"*20)
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == '1':
            print("\nStarting emotion detection with ORIGINAL model...")
            try:
                detector = ModernEmotionDetector(use_finetuned=False)
                detector.run()
            except Exception as e:
                print(f"\nError running emotion detection: {e}")
                input("\nPress Enter to continue...")
        
        elif choice == '2':
            if not has_finetuned:
                print("\nWarning: No fine-tuned model found!")
                print("Please train a model first using option 3.")
                print("Falling back to original model...")
                input("\nPress Enter to continue...")
            
            print("\nStarting emotion detection with FINE-TUNED model...")
            try:
                detector = ModernEmotionDetector(use_finetuned=True)
                detector.run()
            except Exception as e:
                print(f"\nError running emotion detection: {e}")
                input("\nPress Enter to continue...")
        
        elif choice == '3':
            print("\nStarting model training...")
            print("-"*20)
            try:
                import subprocess
                result = subprocess.run([sys.executable, 'src/trained_emotions.py'])
                
                if result.returncode == 0:
                    print("\nTraining completed successfully!")
                else:
                    print("\nTraining exited with errors.")
            except FileNotFoundError:
                print("\nError: trained_emotions.py not found!")
                print("Please ensure the file exists at: src/trained_emotions.py")
            except Exception as e:
                print(f"\nError during training: {e}")
            
            input("\nPress Enter to continue...")
        
        elif choice == '0':
            print("\nExiting... Goodbye!")
            sys.exit(0)
        
        else:
            print("\nInvalid choice. Please select 0, 1, 2, or 3.")
            sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
