# deploy.py

import os
import torch
import gradio as gr
from PIL import Image
from constants import image_transform, AGE_MAP, AGE_MAP_INVERSE
from models import AgeClassifier

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
MODEL_PATH = 'age_predictor.pth'

def load_model(model_path):
    model = AgeClassifier(num_classes=len(AGE_MAP))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH)

# Prediction function
def predict(image):
    try:
        image = image.convert('RGB')
        image = image_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        predicted_label = AGE_MAP_INVERSE[predicted.item()]
        return f"Predicted Age Group: {predicted_label}"
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs="text",
    title="Age Detection AI",
    description="Upload a photo to predict the age group."
)

if __name__ == "__main__":
    iface.launch()
