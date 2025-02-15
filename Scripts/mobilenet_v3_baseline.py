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
wandb.login(key='3043bbf9bb6b869c873214e9e89a66b9894dcaa2')
wandb.init(project="rice-disease-classification")

# Setting Seeds for Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the disease label and image name
        disease = self.df.iloc[idx]['label']
        img_name = self.df.iloc[idx]['image_id']
        
        # Construct the full path by joining img_dir, disease folder, and image name
        img_path = os.path.join(self.img_dir, disease, img_name)
        
        # Load and transform image
        image = Image.open(img_path)
        label = self.df.iloc[idx]['label_encoded']

        if self.transform:
            image = self.transform(image)

        return image, label

def prepare_data():
    # Define paths
    train_path = Path('/Users/ananyashukla/Desktop/Ananya_Shukla/Semester 4/ILGC/low-altitude-drone/paddy-disease-classification/train_images')
    test_path = Path('/Users/ananyashukla/Desktop/Ananya_Shukla/Semester 4/ILGC/low-altitude-drone/paddy-disease-classification/test_images')

    # Load train labels
    train_df = pd.read_csv('/Users/ananyashukla/Desktop/Ananya_Shukla/Semester 4/ILGC/low-altitude-drone/paddy-disease-classification/train.csv')
    
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    
    # Fit and transform the labels
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Split data
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

    # Create datasets
    train_dataset = ImageDataset(df=train_df, img_dir=train_path, transform=transform)
    valid_dataset = ImageDataset(df=valid_df, img_dir=train_path, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    return train_loader, valid_loader, label_encoder, len(np.unique(train_df['label_encoded']))

# Training and Validation
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar for training
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
    
    # Progress bar for validation
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
    # Prepare data
    print('Starting Data Prep')
    train_loader, valid_loader, label_encoder, num_classes = prepare_data()
    print('Data Prep Completed')

    # Initialize model
    print('Defining Model')
    model = mobilenet_v3_small(pretrained=True)
    model = model.to(device)
    print('Model Sent to Device')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    print('Starting Training')
    n_epochs = 50
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Training phase
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Training: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
        
        # Validation phase
        valid_loss, valid_acc, all_true, all_pred = validate(model, valid_loader, criterion, device)
        print(f"Validation: Loss = {valid_loss:.4f}, Accuracy = {valid_acc:.4f}")
        
        # Log metrics to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Valid Loss": valid_loss,
            "Valid Accuracy": valid_acc
        })
    
    # Final evaluation
    class_names = label_encoder.classes_
    all_true, all_pred = validate(model, valid_loader, criterion, device)[2:]
    acc = accuracy_score(all_true, all_pred)
    print("\nMobileNet Model Accuracy on Validation Set: {:.2f}%".format(acc * 100))
    cls_report = classification_report(all_true, all_pred, target_names=class_names, digits=5)
    print(cls_report)

    # Save the model
    torch.save(model.state_dict(), 'mobilenet_v3_model.pth')
    wandb.save('mobilenet_v3_model.pth')

if __name__ == '__main__':
    main()
    wandb.finish()
