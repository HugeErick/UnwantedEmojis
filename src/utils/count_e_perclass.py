#!/usr/bin/env python3
"""
count_e_perclass.py
Quick report of how many images each emotion folder contains.
"""

from pathlib import Path

DATASET_DIR = Path(__file__).with_name("custom_dataset")
IMAGE_EXT = (".jpg", ".jpeg", ".png")

def main():
    if not DATASET_DIR.is_dir():
        print("custom_dataset folder not found")
        return

    totals = []
    for emotion in sorted(DATASET_DIR.iterdir()):
        if not emotion.is_dir():
            continue
        count = sum(1 for f in emotion.iterdir() if f.suffix.lower() in IMAGE_EXT)
        totals.append((emotion.name, count))

    max_label = max(len(name) for name, _ in totals) if totals else 0
    for name, cnt in totals:
        print(f"{name.ljust(max_label)} : {cnt}")

if __name__ == "__main__":
    main()