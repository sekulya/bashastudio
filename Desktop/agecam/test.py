import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from constants import image_transform, AGE_MAP
from models import AgeClassifier
from dataset_loader import AgeGroupDataset

# Import or define AgeGroupDataset here
from dataset_loader import AgeGroupDataset  # Option 1 (recommended)
# OR include the full class definition (Option 2)

# 1. First modify the CSV to create validation split
csv_path = 'C:/Users/Dell/Desktop/agecam/age_detection.csv'
df = pd.read_csv(csv_path)

if 'val' not in df['split'].unique():
    train_df, val_df = train_test_split(
        df[df['split'] == 'train'], 
        test_size=0.2,
        random_state=42
    )
    df.loc[val_df.index, 'split'] = 'val'
    df.to_csv(csv_path, index=False)
    print("Created validation split in CSV")

# 2. Setup device and paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_dir = 'C:/Users/Dell/Desktop/agecam/train'
model_path = 'age_predictor.pth'

# Verify splits
print("Data splits:", df['split'].value_counts())

# Load validation dataset
val_dataset = AgeGroupDataset(csv_path, image_dir, split='val', image_transform=image_transform)
print(f"Found {len(val_dataset)} validation samples")

if len(val_dataset) == 0:
    sample_files = pd.read_csv(csv_path)
    sample_files = sample_files[sample_files['split'] == 'val']['file'].tolist()
    print("Sample validation files:", sample_files[:3])
    print("Image dir contents:", os.listdir(image_dir)[:3])
    exit()

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load and evaluate model
model = AgeClassifier(num_classes=len(AGE_MAP)).to(device)
model.load_state_dict(torch.load(model_path))

def evaluate_model(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

accuracy = evaluate_model(model, val_loader)
print(f"Validation Accuracy: {accuracy:.2f}%")