import torch, torch.nn as nn, os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import ResNetTransformer
from dataset_utils import UnifiedDeepfakeDataset, custom_collate_fn

# ----------------------------------------------------------------------
# 1. HELPER FUNCTION: Processes a single image or video batch
# ----------------------------------------------------------------------
def train_one_batch(model, data_tuple, criterion, optimizer, device):
    inputs, labels = data_tuple
    inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    batch_loss = criterion(outputs, labels)
    
    batch_loss.backward()
    optimizer.step()
    
    loss = batch_loss.item() * inputs.size(0)
    acc = ((torch.sigmoid(outputs) > 0.5) == labels).sum().item()
    total = labels.size(0)
    
    return loss, acc, total

# ----------------------------------------------------------------------
# 2. MAIN TRAINING FUNCTION: Iterates through the dataloader
# ----------------------------------------------------------------------
def train_one_epoch_split(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch, handling image and video batches separately.
    """
    model.train()
    total_loss, total_acc, total_samples = 0.0, 0, 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None: continue
        
        # Train on Image Batch (ResNet path)
        if 'image' in batch:
            loss_img, acc_img, total_img = train_one_batch(model, batch['image'], criterion, optimizer, device)
            total_loss += loss_img
            total_acc += acc_img
            total_samples += total_img

        # Train on Video Batch (ResNet + Transformer path)
        if 'video' in batch:
            loss_vid, acc_vid, total_vid = train_one_batch(model, batch['video'], criterion, optimizer, device)
            total_loss += loss_vid
            total_acc += acc_vid
            total_samples += total_vid
            
    return total_loss / total_samples if total_samples > 0 else 0, \
           total_acc / total_samples if total_samples > 0 else 0


# ----------------------------------------------------------------------
# 3. VALIDATION FUNCTION (Also split for clarity)
# ----------------------------------------------------------------------
def validate_one_batch(model, data_tuple, criterion, device):
    inputs, labels = data_tuple
    inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
    
    outputs = model(inputs)
    
    loss = criterion(outputs, labels).item() * inputs.size(0)
    acc = ((torch.sigmoid(outputs) > 0.5) == labels).sum().item()
    total = labels.size(0)
    
    return loss, acc, total

def validate_one_epoch_split(model, dataloader, criterion, device):
    """
    Validates the model for one epoch, handling image and video batches separately.
    """
    model.eval()
    total_loss, total_acc, total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None: continue
            
            # Validate on Image Batch
            if 'image' in batch:
                loss_img, acc_img, total_img = validate_one_batch(model, batch['image'], criterion, device)
                total_loss += loss_img
                total_acc += acc_img
                total_samples += total_img

            # Validate on Video Batch
            if 'video' in batch:
                loss_vid, acc_vid, total_vid = validate_one_batch(model, batch['video'], criterion, device)
                total_loss += loss_vid
                total_acc += acc_vid
                total_samples += total_vid
            
    return total_loss / total_samples if total_samples > 0 else 0, \
           total_acc / total_samples if total_samples > 0 else 0


# ----------------------------------------------------------------------
# 4. MAIN FUNCTION (Updated to use new split functions)
# ----------------------------------------------------------------------
def main():
    # Note: EPOCHS changed to 1 for brevity, you should set this higher.
    DATA_PATH, EPOCHS, BATCH_SIZE, LR, NUM_FRAMES = './data', 5, 8, 1e-4, 20 
    MODEL_SAVE_PATH = './best_unified_model.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Code to collect files (UNCHANGED) ---
    all_files, all_labels = [], []
    media_types = ['image','video']
    classes = ['real', 'fake']

    for media_type in media_types:
        for cls in classes:
            current_path = os.path.join(DATA_PATH, media_type, cls)
            if os.path.exists(current_path):
                files = [os.path.join(current_path, f) for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]
                labels = [0 if cls == 'real' else 1] * len(files)
                all_files.extend(files)
                all_labels.extend(labels)
    
    files, labels = all_files, all_labels
    # --- End of file collection ---

    train_files, val_files, train_labels, val_labels = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=42)

    train_dataset, val_dataset = UnifiedDeepfakeDataset(train_files, train_labels, NUM_FRAMES), UnifiedDeepfakeDataset(val_files, val_labels, NUM_FRAMES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    model = ResNetTransformer().to(device)
    criterion, optimizer = nn.BCEWithLogitsLoss(), torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # *** CHANGE IS HERE ***
        train_loss, train_acc = train_one_epoch_split(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch_split(model, val_loader, criterion, device)
        # **********************
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_loss > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Saved new best model to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()