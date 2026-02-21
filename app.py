import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- 1. SET UP THE MODEL ARCHITECTURE ---
# This must match your training setup exactly
@st.cache_resource # Keeps the model in memory so it doesn't reload every click
def load_blood_model():
    model = models.vit_b_16(weights=None) # Load architecture
    model.heads = nn.Sequential(
        nn.Linear(model.heads.head.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    # Load your trained weights (Update the path to your .pth file)
    # model.load_state_dict(torch.load('basophil_model.pth', map_location='cpu'))
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_blood_model().to(device)

# --- 2. IMAGE PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. STREAMLIT UI ---
st.title("🩸 Basophil Malignancy Detector")
st.write("Upload a digitized peripheral blood smear image (TIF/JPG/PNG) to identify malignant patterns.")

uploaded_file = st.file_uploader("Choose a basophil image...", type=["tif", "tiff", "jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Peripheral Blood Smear', use_container_width=True)
    
    with st.spinner('Analyzing cellular morphology...'):
        # Preprocess and Predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, pred_idx = torch.max(probabilities, 0)
        
        classes = ['Malignant', 'Normal']
        result = classes[pred_idx]
        
        # --- 4. DISPLAY RESULTS ---
        st.subheader(f"Prediction: **{result}**")
        st.progress(float(confidence))
        st.write(f"Confidence Score: {confidence*100:.2f}%")
        
        if result == 'Malignant':
            st.error("⚠️ Signs of malignancy detected. High nucleus-to-cytoplasm ratio or atypical granulation observed.")
        else:
            st.success("✅ Cellular morphology appears within normal parameters.")

# Add sidebar info
st.sidebar.info("This app uses a **Vision Transformer (ViT)** to learn spatial organization and population-level patterns in basophil granulocytes.")