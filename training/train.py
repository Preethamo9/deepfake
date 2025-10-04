import torch
import torch.nn as nn
import os
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import ResNetTransformer
from dataset_utils import UnifiedDeepfakeDataset, custom_collate_fn
# NEW: Import the learning rate scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, total_samples = 0.0, 0, 0
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None: continue
        
        # This loop correctly handles batches that might contain only images or only videos
        for media_type, (inputs, labels) in batch.items():
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            total_acc += ((torch.sigmoid(outputs) > 0.5) == labels).sum().item()
            total_samples += labels.size(0)
            
    return total_loss / total_samples if total_samples > 0 else 0, total_acc / total_samples if total_samples > 0 else 0

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None: continue
            
            for media_type, (inputs, labels) in batch.items():
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                total_acc += ((torch.sigmoid(outputs) > 0.5) == labels).sum().item()
                total_samples += labels.size(0)
                
    return total_loss / total_samples if total_samples > 0 else 0, total_acc / total_samples if total_samples > 0 else 0

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training on media type: {args.media_type}")
    
    # --- Data Loading ---
    data_dir = os.path.join(args.data_path, args.media_type)
    real_path = os.path.join(data_dir, 'real')
    fake_path = os.path.join(data_dir, 'fake')
    
    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print(f"❌ CRITICAL ERROR: Could not find '{real_path}' or '{fake_path}'.")
        print("Please ensure you have run preprocess.py and the folders exist.")
        return
        
    supported_ext = ['.jpg', '.jpeg', '.png'] # Add video extensions if you preprocess videos
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if any(f.lower().endswith(ext) for ext in supported_ext)]
    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if any(f.lower().endswith(ext) for ext in supported_ext)]
    
    files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)

    if not files:
        print(f"❌ CRITICAL ERROR: No supported files found in '{data_dir}'.")
        return
    
    print(f"Found {len(files)} total files.")
    train_files, val_files, train_labels, val_labels = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=42)

    train_dataset = UnifiedDeepfakeDataset(train_files, train_labels)
    val_dataset = UnifiedDeepfakeDataset(val_files, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    # --- Model, Optimizer, and Scheduler Setup ---
    model = ResNetTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # NEW: Learning rate scheduler reduces LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    start_epoch = 0
    best_val_loss = float('inf')
    model_save_path = f'./best_{args.media_type}_model.pth'

    # NEW: Resume training functionality
    if args.resume and os.path.exists(model_save_path):
        print(f"Resuming training from saved model: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # NEW: Step the scheduler with the validation loss
        scheduler.step(val_loss)
        
        if val_loss > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Saved new best model to {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a deepfake detector.")
    
    # --- Arguments are now used for all key parameters ---
    parser.add_argument('--media_type', type=str, required=True, choices=['image', 'video'], help="Type of media to train on.")
    parser.add_argument('--data_path', type=str, default='./data_processed', help="Path to the preprocessed data directory.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--resume', action='store_true', help="Resume training from the last saved model.")
    
    args = parser.parse_args()
    main(args)