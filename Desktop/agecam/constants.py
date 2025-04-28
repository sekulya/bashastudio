# constants.py
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

AGE_MAP = {
    '18-20': 0, '21-30': 1, '31-40': 2, '41-50': 3,
    '51-60': 4
}

AGE_MAP_INVERSE = {v: k for k, v in AGE_MAP.items()}