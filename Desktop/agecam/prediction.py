import os
import torch
from constants import image_transform, AGE_MAP, AGE_MAP_INVERSE
from models import AgeClassifier
from PIL import Image
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    try:
        model = AgeClassifier(num_classes=len(AGE_MAP)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def predict_age(model: torch.nn.Module, image_path: str, transform, device: torch.device) -> Optional[str]:
    try:
        full_path = os.path.abspath(image_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found at {full_path}")
            
        image = Image.open(full_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        
        return AGE_MAP_INVERSE[predicted.item()]
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Configuration - UPDATE THIS TO YOUR ACTUAL PATH
    MODEL_PATH = 'age_predictor.pth'
    TEST_IMAGE = r"C:\Users\Dell\Desktop\agecam\test\31-40\5.jpg"  # Absolute path

    # Debugging info
    print("\n--- Path Verification ---")
    print(f"Current directory: {os.getcwd()}")
    print(f"Looking for image at: {TEST_IMAGE}")
    print(f"Path exists: {os.path.exists(TEST_IMAGE)}")
    
    if not os.path.exists(TEST_IMAGE):
        print("\nPossible solutions:")
        print("1. Update TEST_IMAGE path in prediction.py")
        print("2. Place image in correct location")
        print(f"Current folder contents: {os.listdir(os.path.dirname(TEST_IMAGE))}")
        exit()

    try:
        print("\nLoading model...")
        model = load_model(MODEL_PATH, device)
        
        print(f"\nPredicting age for {TEST_IMAGE}...")
        result = predict_age(model, TEST_IMAGE, image_transform, device)
        
        print(f"\nPrediction Result: {result if result else 'Failed'}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

