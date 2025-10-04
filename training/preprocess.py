import os
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# --- Configuration ---
# 1. Path to your original dataset with 'real' and 'fake' subfolders
SOURCE_DATA_PATH = './data/image' 

# 2. Path where the processed (cropped face) images will be saved
PROCESSED_DATA_PATH = './data_processed/image' 
# ---------------------

def preprocess_data():
    """
    Detects and crops faces from all images in the source directory
    and saves them to the processed directory.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    
    # MTCNN is used for fast and accurate face detection
    # image_size=224 is a good default for many models like ResNet
    mtcnn = MTCNN(image_size=224, margin=20, device=device, keep_all=False, post_process=False)

    face_found_count = 0
    face_not_found_count = 0

    # Loop through 'real' and 'fake' folders
    for class_name in os.listdir(SOURCE_DATA_PATH):
        source_class_dir = os.path.join(SOURCE_DATA_PATH, class_name)
        processed_class_dir = os.path.join(PROCESSED_DATA_PATH, class_name)

        # Create the destination folder if it doesn't exist
        os.makedirs(processed_class_dir, exist_ok=True)
        
        if not os.path.isdir(source_class_dir):
            continue

        image_files = [f for f in os.listdir(source_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f'\nProcessing class: {class_name}...')
        for file_name in tqdm(image_files, desc=f'Processing {class_name}'):
            source_file_path = os.path.join(source_class_dir, file_name)
            processed_file_path = os.path.join(processed_class_dir, file_name)
            
            # Skip if the file has already been processed
            if os.path.exists(processed_file_path):
                continue
            
            try:
                img = Image.open(source_file_path).convert('RGB')
                
                # Use MTCNN's save method for efficiency
                # It detects, crops, resizes, and saves the image in one go.
                mtcnn(img, save_path=processed_file_path)

                if os.path.exists(processed_file_path):
                    face_found_count += 1
                else:
                    # mtcnn returns None if no face is found, so save_path won't be created
                    face_not_found_count += 1

            except Exception as e:
                print(f"Error processing {source_file_path}: {e}")
                face_not_found_count += 1

    print("\n--- Pre-processing Complete ---")
    print(f"Total faces found and saved: {face_found_count}")
    print(f"Total images where no face was found: {face_not_found_count}")

if __name__ == '__main__':
    preprocess_data()