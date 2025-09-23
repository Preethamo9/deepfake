import torch, cv2, numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image

class UnifiedDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, num_frames):
        self.file_paths, self.labels, self.num_frames = file_paths, labels, num_frames
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(image_size=224, margin=20, device=self.device, keep_all=False, post_process=False)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self): return len(self.file_paths)
    def _process_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            face_tensor = self.mtcnn(img)
            if face_tensor is None: return None
            return self.transform((face_tensor + 1) / 2)
        except Exception: return None
    def _process_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1: return None
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        for i in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames: return None
        try:
            faces = self.mtcnn(frames)
            if faces is None or len(faces) == 0: return None
            face_frames = [transforms.ToPILImage()((f + 1) / 2) for f in faces if f is not None]
            if len(face_frames) < self.num_frames:
                if not face_frames: return None
                face_frames.extend([face_frames[-1]] * (self.num_frames - len(face_frames)))
            return torch.stack([self.transform(f) for f in face_frames])
        except Exception: return None
    def __getitem__(self, idx):
        path, label = self.file_paths[idx], self.labels[idx]
        is_video = any(path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
        data = self._process_video(path) if is_video else self._process_image(path)
        if data is None: return None
        return data, torch.tensor(label, dtype=torch.float32), 'video' if is_video else 'image'

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    images, videos, image_labels, video_labels = [], [], [], []
    for data, label, media_type in batch:
        if media_type == 'image':
            images.append(data); image_labels.append(label)
        else:
            videos.append(data); video_labels.append(label)
    processed_batch = {}
    if images: processed_batch['image'] = (torch.stack(images), torch.stack(image_labels))
    if videos: processed_batch['video'] = (torch.stack(videos), torch.stack(video_labels))
    return processed_batch