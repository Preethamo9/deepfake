import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# This dataset is MUCH faster because it assumes faces have already been
# detected and cropped by the preprocess.py script.
class UnifiedDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, num_frames=None): # num_frames is kept for consistency but not used for images
        self.file_paths = file_paths
        self.labels = labels
        # The only transformation needed is to convert to tensor and normalize.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # We just load the pre-cropped image directly. No MTCNN needed!
            img = Image.open(path).convert('RGB')
            processed_data = self.transform(img)
            # We assume all data in the processed folder is an 'image' (a cropped face)
            media_type = 'image' 
            return processed_data, torch.tensor(label, dtype=torch.float32), media_type
        except Exception as e:
            print(f"Warning: Could not load image {path}. Skipping. Error: {e}")
            return None

# NO CHANGES NEEDED HERE. This function is still perfect for sorting
# the batches into images and videos (if you add videos later).
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    images, videos, image_labels, video_labels = [], [], [], []
    for data, label, media_type in batch:
        if media_type == 'image':
            images.append(data)
            image_labels.append(label)
        else: # For videos
            videos.append(data)
            video_labels.append(label)

    processed_batch = {}
    if images:
        processed_batch['image'] = (torch.stack(images), torch.stack(image_labels))
    if videos:
        processed_batch['video'] = (torch.stack(videos), torch.stack(video_labels))
        
    return processed_batch