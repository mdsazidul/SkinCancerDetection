from google.colab import drive
drive.mount('/content/drive')
#trrained on colab pro plus with gpu A100

import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
import torch
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from torchvision.transforms import RandAugment


# 1. CONFIGURATION

data_dir = '/content/drive/MyDrive/skin_cancer_detection_app/swin_skin_dataset'
batch_size = 4               # smaller batch for Colab memory
accumulation_steps = 6       # effective batch size = 4 × 6 = 24
image_size = 384
epochs = 50
num_classes = 7
patience = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Device: {device}")


# 3. DATA AUGMENTATION

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15),
                            scale=(0.85, 1.15), shear=10),
    RandAugment(num_ops=3, magnitude=9),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.15),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 4. DATASETS & LOADERS

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)

# Compute class distribution
class_counts = [0] * num_classes
for _, label in train_dataset:
    class_counts[label] += 1

print("="*60)
print("CLASS DISTRIBUTION:")
total = sum(class_counts)
for i, name in enumerate(train_dataset.classes):
    print(f"{name:10s}: {class_counts[i]:5d} ({100*class_counts[i]/total:.2f}%)")
print("="*60)

# Weighted sampler for imbalance
max_count = max(class_counts)
sample_weights = [max_count / class_counts[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)


# 5. MODEL

model = timm.create_model(
    'swin_base_patch4_window12_384',
    pretrained=True,
    num_classes=num_classes,
    drop_rate=0.3,
    drop_path_rate=0.2
)

# Enable gradient checkpointing to save memory
model.set_grad_checkpointing(True)

class ModelWithUncertainty(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, x, training=False):
        return self.base_model(x)
    def predict_with_uncertainty(self, x, num_samples=3):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                p = torch.softmax(self.base_model(x), dim=1)
                preds.append(p)
        preds = torch.stack(preds)
        return preds.mean(0), preds.std(0)

model = ModelWithUncertainty(model).to(device)
print(f"✅ Model: Swin-Base-384 ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")


# 6. LOSS FUNCTION

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets,
                                         weight=self.weight,
                                         reduction='none',
                                         label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()

class_weights = torch.FloatTensor([max(class_counts) / c for c in class_counts]).to(device)
criterion = FocalLoss(gamma=2.5, weight=class_weights, label_smoothing=0.1)
print(f"⚖️ Class Weights: {class_weights.cpu().numpy()}")


# 7. OPTIMIZER & SCHEDULER

for name, param in model.named_parameters():
    if 'base_model.layers.0' in name or 'base_model.layers.1' in name:
        param.requires_grad = False

param_groups = [
    {'params': [p for n, p in model.named_parameters() if 'head' in n], 'lr': 5e-4, 'weight_decay': 1e-2},
    {'params': [p for n, p in model.named_parameters() if 'head' not in n and p.requires_grad], 'lr': 1e-5, 'weight_decay': 1e-3}
]
optimizer = optim.AdamW(param_groups)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
scaler = torch.amp.GradScaler('cuda')


# 8. EARLY STOPPING

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

early_stopping = EarlyStopping(patience=patience)


# 9. TRAINING LOOP

best_val_acc = 0
train_losses, val_losses, val_accuracies = [], [], []

print("\n🚀 STARTING TRAINING\n")
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for step, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(imgs, training=True)
            loss = criterion(outputs, labels) / accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({'loss': f'{loss.item()*accumulation_steps:.4f}',
                          'acc': f'{100*correct/total:.2f}%'})

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)

    #  VALIDATION
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

    # Save  model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'class_names': train_dataset.classes
        }, '/content/drive/MyDrive/skin_cancer_detection_app/best_model.pth')
        print(f"✅ Saved new best model (Val Acc: {val_acc:.2f}%)")

    scheduler.step()
    if epoch == 10:
        print("🔓 Unfreezing early layers...")
        for p in model.parameters():
            p.requires_grad = True

    if early_stopping(val_loss):
        print(f"⏹️ Early stopping at epoch {epoch+1}")
        break

    torch.cuda.empty_cache()
    print("-"*60)

print(f" Training complete! Best Val Acc = {best_val_acc:.2f}%")

