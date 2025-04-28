# Install required packages
# Run this in your terminal or notebook cell if not already installed:
# !pip3 install numpy torch torchvision torchaudio pandas Pillow
# dataset_loader.py
# dataset_loader.py

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from constants import image_transform, AGE_MAP, AGE_MAP_INVERSE
from models import AgeClassifier
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AgeGroupDataset(Dataset):
    def __init__(self, csv_path, image_dir, split='train', image_transform=None):
        self.image_dir = image_dir
        self.image_transform = image_transform  # ✅ Corrected here
        self.age_map = AGE_MAP

        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split]
        self.df = self.df.dropna(subset=['file', 'age'])
        self.df['label'] = self.df['age'].map(self.age_map)
        self.df = self.df.dropna(subset=['label'])
        self.df['label'] = self.df['label'].astype(int)

        self.image_paths = self.df['file'].tolist()
        self.labels = self.df['label'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_file = self.image_paths[idx].replace("train/", "").replace("test/", "")
        img_path = os.path.join(self.image_dir, img_file)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.image_transform:
            image = self.image_transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def train_model(model, dataloader, criterion, optimizer, scheduler, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, ages in dataloader:
            images, ages = images.to(device), ages.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    return model

def main():
    csv_path = 'C:/Users/Dell/Desktop/agecam/age_detection.csv'
    image_dir = 'C:/Users/Dell/Desktop/agecam/train'

    try:
        dataset = AgeGroupDataset(csv_path, image_dir, image_transform=image_transform)  # ✅ Corrected here too
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

        model = AgeClassifier(num_classes=len(AGE_MAP)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        print("\nStarting training...")
        trained_model = train_model(model, dataloader, criterion, optimizer, scheduler, epochs=10)

        torch.save(trained_model.state_dict(), 'age_predictor.pth')
        print("Model saved successfully!")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Check the following:")
        print(f"1. CSV path: {csv_path}")
        print(f"2. Image directory: {image_dir}")
        print(f"3. File permissions and contents")

if __name__ == "__main__":
    main()
