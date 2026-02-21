import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 
from PIL import Image
from torchvision import transforms, models

# --- 1. CNN ARCHITECTURE DEFINITION ---
class BloodCellCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BloodCellCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x1 = F.relu(self.bn3(self.conv3(x))) # Feature map layer
        x = self.pool(x1)
        x_flat = x.view(-1, 128 * 28 * 28)
        x_out = F.relu(self.fc1(x_flat))
        logits = self.fc2(self.dropout(x_out))
        return logits, x1

# --- 2. HELPER FUNCTIONS ---

@st.cache_resource
def load_cnn_model():
    model = BloodCellCNN(num_classes=2)
    # Ensure 'cnn_blood_model.pth' exists in your directory
    try:
        model.load_state_dict(torch.load('cnn_blood_model.pth', map_location='cpu'))
    except:
        st.sidebar.warning("Weights 'cnn_blood_model.pth' not found. Using untrained weights.")
    model.eval()
    return model

def segment_nucleus(image_pil):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 40, 40])
    upper_purple = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_cnt)
        pad = 20
        crop = img_cv[max(0, y-pad):min(img_cv.shape[0], y+h+pad), 
                      max(0, x-pad):min(img_cv.shape[1], x+w+pad)]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), True
    return image_pil, False

def get_nc_ratio(image_pil):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, cell_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cell_area = cv2.countNonZero(cell_mask)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    nuc_mask = cv2.inRange(hsv, np.array([120, 40, 40]), np.array([160, 255, 255]))
    nuc_area = cv2.countNonZero(nuc_mask)
    ratio = (nuc_area / cell_area) * 100 if cell_area > 0 else 0
    return ratio, nuc_mask

# --- 3. STREAMLIT UI ---

st.set_page_config(page_title="Hematology AI Lab", layout="wide")
st.title("🔬 Malignant blood disorders")

model = load_cnn_model()

st.sidebar.header("Analysis Options")
show_nc = st.sidebar.checkbox("Calculate N:C Ratio", value=True)
show_features = st.sidebar.checkbox("Show CNN Feature Extraction", value=True)

uploaded_file = st.file_uploader("Upload Basophil Smear...", type=["tif", "jpg", "png"])

if uploaded_file:
    raw_img = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Input Image")
        st.image(raw_img, use_container_width=True)
        
    nucleus_img, found = segment_nucleus(raw_img)
    
    with col2:
        st.write("### Segmented Nucleus")
        st.image(nucleus_img, use_container_width=True)

    # --- 4. PREDICTION & FEATURE EXTRACTION ---
    st.divider()
    
    # Preprocess the cropped nucleus for the CNN
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(nucleus_img).unsqueeze(0)

    with torch.no_grad():
        logits, feature_maps = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    # Display Prediction Results
    res_label = "Malignant" if pred == 0 else "Normal"
    st.subheader(f"Prediction: **{res_label}**")
    st.write(f"Confidence: **{conf.item()*100:.2f}%**")

    # --- 5. VISUAL & MORPHOMETRIC ADD-ONS ---
    col_a, col_b = st.columns(2)

    if show_features:
        with col_a:
            st.write("### CNN Feature Map")
            # Create a heatmap from the last conv layer
            heatmap = torch.mean(feature_maps[0], dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            st.image(heatmap, caption="High-activation regions (Chromatin pattern)", use_container_width=True)

    if show_nc:
        with col_b:
            ratio, mask = get_nc_ratio(raw_img)
            st.write("### Morphometrics")
            st.metric("N:C Ratio", f"{ratio:.1f}%")
            if ratio > 70:
                st.error("⚠️ Abnormal N:C Ratio detected.")
            else:
                st.success("✅ Normal nuclear proportions.")
