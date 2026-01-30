"""
Download CS:GO Dataset from Roboflow
====================================
Downloads a pre-labeled Counter-Strike dataset for training.

Usage:
1. Get your API key from https://app.roboflow.com/ (Settings > API Key)
2. Run this script
3. Enter your API key when prompted
"""

from roboflow import Roboflow
import os


def main():
    print("=" * 60)
    print("ROBOFLOW DATASET DOWNLOAD")
    print("=" * 60)
    print()
    print("This will download the CS:GO player detection dataset.")
    print("You need a free Roboflow API key.")
    print()
    print("Get your key at: https://app.roboflow.com/")
    print("  → Click profile icon → Settings → API Key")
    print()
    print("=" * 60)

    # Get API key from user
    api_key = input("\nEnter your Roboflow API key: ").strip()

    if not api_key:
        print("Error: No API key provided.")
        return

    try:
        print("\nConnecting to Roboflow...")
        rf = Roboflow(api_key=api_key)

        print("Accessing CS:GO dataset...")
        # This is the Counter Strike Global Offensive dataset
        project = rf.workspace("bajs-macka").project("counter-strike-global-offensive")

        print("Downloading dataset (YOLOv8 format)...")
        # Download version 5 (latest with most images)
        dataset = project.version(5).download("yolov8", location="data/csgo-dataset")

        print()
        print("=" * 60)
        print("SUCCESS! Dataset downloaded to: data/csgo-dataset")
        print("=" * 60)
        print()
        print("Dataset structure:")
        print("  data/csgo-dataset/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  ├── valid/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  └── data.yaml")
        print()
        print("Next step: Run train.py to train the model!")

    except Exception as e:
        print(f"\nError: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure your API key is correct")
        print("2. Check your internet connection")
        print("3. Try downloading manually from:")
        print("   https://universe.roboflow.com/bajs-macka/counter-strike-global-offensive")


if __name__ == "__main__":
    main()
