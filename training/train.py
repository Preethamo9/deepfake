import torch, torch.nn as nn, os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import ResNetTransformer
from dataset_utils import UnifiedDeepfakeDataset, custom_collate_fn

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss, acc, total = 0.0, 0, 0
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None: continue
        for media_type in ['image', 'video']:
            if media_type in batch:
                inputs, labels = batch[media_type]
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                batch_loss.backward()
                optimizer.step()
                loss += batch_loss.item() * inputs.size(0)
                acc += ((torch.sigmoid(outputs) > 0.5) == labels).sum().item()
                total += labels.size(0)
    return loss / total if total > 0 else 0, acc / total if total > 0 else 0

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    loss, acc, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None: continue
            for media_type in ['image', 'video']:
                if media_type in batch:
                    inputs, labels = batch[media_type]
                    inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(inputs)
                    loss += criterion(outputs, labels).item() * inputs.size(0)
                    acc += ((torch.sigmoid(outputs) > 0.5) == labels).sum().item()
                    total += labels.size(0)
    return loss / total if total > 0 else 0, acc / total if total > 0 else 0

def main():
    DATA_PATH, EPOCHS, BATCH_SIZE, LR, NUM_FRAMES = './data', 10, 8, 1e-4, 20
    MODEL_SAVE_PATH = './best_unified_model.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    real_path, fake_path = os.path.join(DATA_PATH, 'real'), os.path.join(DATA_PATH, 'fake')
    supported_ext = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov']
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if any(f.lower().endswith(ext) for ext in supported_ext)]
    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if any(f.lower().endswith(ext) for ext in supported_ext)]
    files, labels = real_files + fake_files, [0] * len(real_files) + [1] * len(fake_files)
    train_files, val_files, train_labels, val_labels = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=42)

    train_dataset, val_dataset = UnifiedDeepfakeDataset(train_files, train_labels, NUM_FRAMES), UnifiedDeepfakeDataset(val_files, val_labels, NUM_FRAMES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    model = ResNetTransformer().to(device)
    criterion, optimizer = nn.BCEWithLogitsLoss(), torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_loss > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved new best model to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()