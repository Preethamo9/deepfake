import torch
import os
from dataset_utils import UnifiedDeepfakeDataset  # Imports the class from your other file

# --- CONFIGURATION ---
# Make sure these paths match your project structure
# This assumes you run this script from the 'training' folder
DATA_DIR = '../data' 
NUM_FRAMES = 20

print("--- Starting Dataset Debugger ---")

# --- 1. Check if the data folders exist ---
real_path = os.path.join(DATA_DIR, 'real')
fake_path = os.path.join(DATA_DIR, 'fake')

print(f"Checking for real folder at: {os.path.abspath(real_path)}")
if not os.path.exists(real_path):
    print("❌ ERROR: 'real' folder not found!")
else:
    print(f"✅ Found 'real' folder with {len(os.listdir(real_path))} files.")

print(f"Checking for fake folder at: {os.path.abspath(fake_path)}")
if not os.path.exists(fake_path):
    print("❌ ERROR: 'fake' folder not found!")
else:
    print(f"✅ Found 'fake' folder with {len(os.listdir(fake_path))} files.")

# --- 2. Try to load the first item from the dataset ---
supported_ext = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
all_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if any(f.lower().endswith(ext) for ext in supported_ext)]
all_files += [os.path.join(fake_path, f) for f in os.listdir(fake_path) if any(f.lower().endswith(ext) for ext in supported_ext)]

if not all_files:
    print("\n❌ CRITICAL ERROR: No supported image or video files found in data folders!")
else:
    print(f"\nFound {len(all_files)} total media files. Attempting to load the first one...")
    
    # We create a dummy dataset with just one file to test it
    test_dataset = UnifiedDeepfakeDataset(
        file_paths=[all_files[0]], 
        labels=[0], 
        num_frames=NUM_FRAMES
    )
    
    # Try to get the item
    try:
        item = test_dataset[0]
        
        if item is None:
            print("\n❌ FAILED: The dataset loaded the file path, but failed to process it.")
            print("   This is likely a FACE DETECTION FAILURE. The MTCNN could not find a face in the media file.")
            print(f"   File path: {all_files[0]}")
        else:
            data, label, media_type = item
            print("\n✅ SUCCESS: Successfully loaded and processed the first media file.")
            print(f"   - Media Type: {media_type}")
            print(f"   - Label: {label.item()}")
            print(f"   - Output Tensor Shape: {data.shape}")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: An unexpected error occurred during data loading: {e}")

print("\n--- Debugging Finished ---")