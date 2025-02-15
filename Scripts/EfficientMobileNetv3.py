# Importing Libraries
import os
import random
import numpy as np
import pandas as pd
import wandb
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from pathlib import Path

# Initialize wandb
wandb.login(key='wandb_key')
wandb.init(project="project", name = "name")

# Setting Seeds for Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        reduced_dim = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale

# Global Attentive Pooling
class GlobalAttentivePooling(nn.Module):
    def __init__(self, in_channels):
        super(GlobalAttentivePooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        weighted_sum = (x * attention_weights).sum(dim=(2, 3))
        normalization_factor = attention_weights.sum(dim=(2, 3)) + 1e-6
        return weighted_sum / normalization_factor

# Multi-Scale Feature Fusion Block
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, num_scales=3):
        super(MultiScaleFeatureFusion, self).__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2**i, dilation=2**i, groups=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                # Changed LayerNorm to handle 4D tensors properly
                nn.GroupNorm(8, in_channels)  # Using GroupNorm instead of LayerNorm
            ) for i in range(num_scales)
        ])

    def forward(self, x):
        scale_features = [scale(x) for scale in self.scales]
        return torch.cat(scale_features, dim=1)

# Optimized MobileNetV3 Small
class OptimizedMobileNetV3Small(nn.Module):
    def __init__(self, num_classes=10, width_multiplier=0.75):
        super(OptimizedMobileNetV3Small, self).__init__()
        base_model = mobilenet_v3_small(pretrained=True)
        feature_dim = int(576 * width_multiplier)

        self.features = base_model.features
        last_conv = nn.Conv2d(
            in_channels=96,
            out_channels=feature_dim,
            kernel_size=1
        )
        self.features[-1] = nn.Sequential(last_conv, SEBlock(feature_dim))
        self.feature_fusion = MultiScaleFeatureFusion(in_channels=feature_dim, num_scales=3)
        self.global_pool = GlobalAttentivePooling(in_channels=feature_dim * 3)

        # Modified classifier to handle the 4D tensor properly
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim * 3),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 3, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.feature_fusion(x)  # Output shape: [batch_size, feature_dim*3, height, width]
        x = self.classifier(x)  # Now properly handles the 4D tensor
        return x

# Custom Loss (Focal Loss + Label Smoothing)
class CustomDiseaseLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0, alpha=0.25, label_smoothing=0.1):
        super(CustomDiseaseLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = label_smoothing

    def forward(self, logits, targets):
        with torch.no_grad():
            smooth_labels = torch.full_like(logits, self.smoothing / (self.num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1))
        focal_term = self.alpha * (1 - pt) ** self.gamma

        loss = -focal_term * torch.log(pt + 1e-6)
        smoothed_loss = -torch.sum(smooth_labels * torch.log(probs + 1e-6), dim=1)

        return (loss + smoothed_loss).mean()
    
# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        disease = self.df.iloc[idx]['label']
        img_name = self.df.iloc[idx]['image_id']
        
        img_path = os.path.join(self.img_dir, disease, img_name)
        
        image = Image.open(img_path)
        label = self.df.iloc[idx]['label_encoded']

        if self.transform:
            image = self.transform(image)

        return image, label

def prepare_data():
    train_path = Path('train_images')
    test_path = Path('test_images')
    train_df = pd.read_csv('train.csv')
    
    # Label Encoding
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.RandomHorizontalFlip(p = 0.4),
        transforms.RandomRotation(degrees = 20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
    train_dataset = ImageDataset(df=train_df, img_dir=train_path, transform=transform)
    valid_dataset = ImageDataset(df=valid_df, img_dir=train_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    return train_loader, valid_loader, label_encoder, len(np.unique(train_df['label_encoded']))

# Training and Validation
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False, unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}', 
            'Accuracy': f'{(predicted == labels).sum().item() / labels.size(0):.4f}'
        })

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    return train_loss, train_acc

def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_pred = []
    all_true = []
    
    pbar = tqdm(valid_loader, desc='Validating', leave=False, unit="batch")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_pred.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Accuracy': f'{(predicted == labels).sum().item() / labels.size(0):.4f}'
            })

    valid_loss = running_loss / len(valid_loader)
    valid_acc = correct / total
    return valid_loss, valid_acc, all_true, all_pred

def main():
    print('Starting Data Prep')
    train_loader, valid_loader, label_encoder, num_classes = prepare_data()
    print('Data Prep Completed')

    # Initializing Model
    print('Defining Model')
    model = OptimizedMobileNetV3Small(num_classes=num_classes, width_multiplier=0.75)
    model = model.to(device)
    print('Model Sent to Device')

    # Loss and Optimizer
    criterion = CustomDiseaseLoss(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training Loop
    print('Starting Training')
    n_epochs = 50
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Training: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
        
        valid_loss, valid_acc, all_true, all_pred = validate(model, valid_loader, criterion, device)
        print(f"Validation: Loss = {valid_loss:.4f}, Accuracy = {valid_acc:.4f}")
        
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Valid Loss": valid_loss,
            "Valid Accuracy": valid_acc
        })
    
    # Evaluation
    class_names = label_encoder.classes_
    all_true, all_pred = validate(model, valid_loader, criterion, device)[2:]
    acc = accuracy_score(all_true, all_pred)
    print("\nMobileNet Model Accuracy on Validation Set: {:.2f}%".format(acc * 100))
    cls_report = classification_report(all_true, all_pred, target_names=class_names, digits=5)
    print(cls_report)

    # Saving Model
    torch.save(model.state_dict(), 'mobilenet_v3_model.pth')
    wandb.save('mobilenet_v3_model.pth')
    
if __name__ == '__main__':
    main()
    wandb.finish()