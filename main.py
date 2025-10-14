import sys
from emotions import ModernEmotionDetector

def main():
    while True:
        print("\n==== Main Menu ====")
        print("1. Run Emotion Detection")
        print("2. Train Model Further")
        print("0. Exit")
        choice = input("Select an option: ").strip()
        if choice == '1':
            detector = ModernEmotionDetector()
            detector.run()
        elif choice == '2':
            import subprocess
            subprocess.run([sys.executable, 'trained_emotions.py'])
        elif choice == '0':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
