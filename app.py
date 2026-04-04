import io
import base64
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from google import genai
from google.genai import types as genai_types

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hematology AI Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .report-box {
        background: #0e1117;
        border: 1px solid #2d2d2d;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ── 1. CNN ARCHITECTURE ───────────────────────────────────────────────────────
class BloodCellCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1);  self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1);self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1  = nn.Linear(128 * 28 * 28, 512)
        self.drop = nn.Dropout(0.5)
        self.fc2  = nn.Linear(512, num_classes)

    def forward(self, x):
        x  = self.pool(F.relu(self.bn1(self.conv1(x))))
        x  = self.pool(F.relu(self.bn2(self.conv2(x))))
        x1 = F.relu(self.bn3(self.conv3(x)))
        x  = self.pool(x1)
        x  = x.view(-1, 128 * 28 * 28)
        x  = F.relu(self.fc1(x))
        return self.fc2(self.drop(x)), x1

# ── 2. HELPERS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    m = BloodCellCNN(num_classes=2)
    try:
        m.load_state_dict(torch.load("cnn_blood_model.pth", map_location="cpu"))
    except Exception:
        pass
    m.eval()
    return m

def segment_nucleus(image_pil):
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([120,40,40]), np.array([160,255,255]))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        p = 20
        crop = img[max(0,y-p):min(img.shape[0],y+h+p),
                   max(0,x-p):min(img.shape[1],x+w+p)]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), True
    return image_pil, False

def get_nc_ratio(image_pil):
    img  = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, cell_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    nuc_mask = cv2.inRange(hsv, np.array([120,40,40]), np.array([160,255,255]))
    cell_area = cv2.countNonZero(cell_mask)
    nuc_area  = cv2.countNonZero(nuc_mask)
    ratio = (nuc_area / cell_area * 100) if cell_area > 0 else 0
    return ratio, nuc_mask

def overlay_mask(image_pil, mask_gray, color=(255, 80, 80), alpha=0.45):
    base  = np.array(image_pil.convert("RGB"))
    color_layer = np.zeros_like(base)
    color_layer[mask_gray > 0] = color
    blended = np.where(mask_gray[:,:,None] > 0,
                       (base * (1 - alpha) + color_layer * alpha).astype(np.uint8),
                       base)
    return Image.fromarray(blended)

def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

def gemini_vision_report(client, image_pil: Image.Image,
                          label: str, confidence: float, nc_ratio: float) -> str:
    buf = io.BytesIO()
    image_pil.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()

    prompt = (
        f"This is a Wright-Giemsa stained basophil blood smear.\n"
        f"A CNN classifier predicted: **{label}** with {confidence:.1f}% confidence.\n"
        f"Morphometric N:C ratio: {nc_ratio:.1f}%.\n\n"
        "Please provide a structured clinical interpretation covering:\n"
        "1. Visible morphological features (nucleus shape, chromatin, granularity)\n"
        "2. Whether the staining pattern supports or conflicts with the CNN prediction\n"
        "3. Key differentials to consider\n"
        "4. Recommended follow-up tests\n\n"
        "Keep the report concise (≤250 words) and clinically actionable."
    )
    response = client.models.generate_content(
        model="models/gemini-1.5-flash-latest",
        contents=[
            genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )
    return response.text

# ── 3. SIDEBAR ────────────────────────────────────────────────────────────────
api_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.title("Hematology AI Lab")
    st.markdown("AI powered by **Google Gemini**")
    if not api_key:
        st.error("API key missing. Add it to `.streamlit/secrets.toml`.")
    else:
        st.success("API key loaded.")
    st.divider()
    st.header("Analysis Options")
    show_nc       = st.checkbox("N:C Ratio", value=True)
    show_overlay  = st.checkbox("N:C Mask Overlay", value=True)
    show_features = st.checkbox("CNN Feature Map", value=True)
    show_report   = st.checkbox("AI Morphology Report", value=True)

# ── 4. TABS ───────────────────────────────────────────────────────────────────
tab_analysis, tab_chat = st.tabs(["🔬 Image Analysis", "💬 Hematology Assistant"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    cnn_model = load_cnn_model()

    uploaded = st.file_uploader("Upload a basophil smear (TIF / JPG / PNG)",
                                 type=["tif","jpg","png"])

    if uploaded:
        raw_img = Image.open(uploaded).convert("RGB")
        nucleus_img, found = segment_nucleus(raw_img)

        # Row 1: images
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Input Image**")
            st.image(raw_img, width="stretch")
        with c2:
            st.markdown("**Segmented Nucleus**")
            st.image(nucleus_img, width="stretch")
            if not found:
                st.caption("No purple nucleus detected — using full image.")

        st.divider()

        # CNN inference
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        tensor = tfm(nucleus_img).unsqueeze(0)
        with torch.no_grad():
            logits, fmaps = cnn_model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred]) * 100

        label = "Malignant" if pred == 0 else "Normal"

        # Key metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", label)
        m2.metric("CNN Confidence", f"{conf:.1f}%")

        nc_ratio, nc_mask = (0, None)
        if show_nc:
            nc_ratio, nc_mask = get_nc_ratio(raw_img)
            m3.metric("N:C Ratio", f"{nc_ratio:.1f}%",
                      delta="Abnormal" if nc_ratio > 70 else "Normal",
                      delta_color="inverse")

        # Probability bar chart
        import pandas as pd
        st.markdown("**Class Probabilities**")
        st.bar_chart(
            pd.DataFrame({"Probability": [float(probs[0])*100, float(probs[1])*100]},
                         index=["Malignant", "Normal"]),
            height=160
        )

        st.divider()

        # Visual add-ons
        vis_cols = []
        if show_features: vis_cols.append("feat")
        if show_nc and show_overlay: vis_cols.append("overlay")

        if vis_cols:
            cols = st.columns(len(vis_cols))
            idx = 0
            if show_features:
                with cols[idx]:
                    st.markdown("**CNN Feature Map**")
                    heatmap = torch.mean(fmaps[0], dim=0).cpu().numpy()
                    heatmap = np.maximum(heatmap, 0)
                    if heatmap.max() > 0:
                        heatmap /= heatmap.max()
                    st.image(heatmap, caption="Chromatin activation regions", width="stretch")
                idx += 1
            if show_nc and show_overlay and nc_mask is not None:
                with cols[idx]:
                    st.markdown("**N:C Mask Overlay**")
                    st.image(overlay_mask(raw_img, nc_mask),
                             caption="Nuclear region highlighted", width="stretch")

        # AI Morphology Report
        if show_report:
            st.divider()
            st.markdown("**AI Morphology Report (Gemini Vision)**")
            if not api_key:
                st.info("Enter your free Gemini API key in the sidebar to generate an AI report.")
            else:
                if st.button("Generate AI Report", type="primary"):
                    with st.spinner("Gemini is analysing the smear…"):
                        try:
                            gclient = get_gemini_client(api_key)
                            report = gemini_vision_report(gclient, raw_img, label, conf, nc_ratio)
                            st.session_state["last_report"] = report
                            st.session_state["last_analysis"] = {
                                "label": label, "confidence": conf, "nc_ratio": nc_ratio,
                            }
                        except Exception as e:
                            st.error(f"Error: {e}")

                if "last_report" in st.session_state:
                    st.markdown(
                        f'<div class="report-box">{st.session_state["last_report"]}</div>',
                        unsafe_allow_html=True)
                    if st.button("💬 Discuss this case in the chatbot"):
                        analysis = st.session_state.get("last_analysis", {})
                        primer = (
                            f"I analysed a basophil smear. CNN predicted **{analysis.get('label','?')}** "
                            f"with {analysis.get('confidence',0):.1f}% confidence. "
                            f"N:C ratio: {analysis.get('nc_ratio',0):.1f}%.\n\n"
                            f"AI morphology report:\n{st.session_state['last_report']}\n\n"
                            "Please help me understand these findings."
                        )
                        st.session_state["messages"] = [{"role": "user", "parts": [primer]}]
                        st.session_state["pending_response"] = True
                        st.info("Switch to the **Hematology Assistant** tab to continue.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("Hematology Assistant")
    st.caption("Ask about blood disorders, CBC interpretation, basophil morphology, and more.")

    SYSTEM_PROMPT = (
        "You are an expert hematology AI assistant specialising in blood disorders and laboratory medicine. "
        "You help clinicians, researchers, and students understand basophil biology, leukemias, lymphomas, "
        "CBC interpretation, bone marrow pathology, flow cytometry, and morphological analysis. "
        "Be concise, accurate, and evidence-based. Acknowledge uncertainty and recommend specialist review "
        "for definitive diagnoses. Format responses with clear headings when helpful."
    )

    QUICK_QUESTIONS = [
        "What features distinguish malignant basophils?",
        "How is N:C ratio used to assess malignancy?",
        "What are WHO criteria for basophilia?",
        "Which CBC findings suggest CML?",
        "Best stains for basophil granule identification?",
    ]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Auto-respond to a case primer from Image Analysis tab
    if st.session_state.get("pending_response") and api_key:
        st.session_state["pending_response"] = False
        msgs = st.session_state.messages
        if msgs and msgs[-1]["role"] == "user":
            try:
                gclient = get_gemini_client(api_key)
                response = gclient.models.generate_content(
                    model="models/gemini-1.5-flash-latest",
                    contents=msgs[-1]["parts"][0],
                    config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                )
                msgs.append({"role": "model", "parts": [response.text]})
            except Exception as e:
                msgs.append({"role": "model", "parts": [f"Error: {e}"]})

    # Quick-question chips (only when chat is empty)
    if not st.session_state.messages:
        st.markdown("**Suggested questions:**")
        cols = st.columns(len(QUICK_QUESTIONS))
        for i, q in enumerate(QUICK_QUESTIONS):
            if cols[i].button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "parts": [q]})
                st.rerun()

    # Render conversation history
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["parts"][0])

    # Chat input
    if prompt := st.chat_input("Ask about hematology…"):
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            ph = st.empty()

            if not api_key:
                ph.warning("Enter your free Gemini API key in the sidebar.")
                st.session_state.messages.pop()
                st.stop()

            try:
                gclient = get_gemini_client(api_key)
                # Build contents from full history
                contents = []
                for m in st.session_state.messages:
                    role = "user" if m["role"] == "user" else "model"
                    contents.append(genai_types.Content(
                        role=role,
                        parts=[genai_types.Part.from_text(text=m["parts"][0])]
                    ))
                response = gclient.models.generate_content(
                    model="models/gemini-1.5-flash-latest",
                    contents=contents,
                    config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                )
                reply_text = response.text
                ph.markdown(reply_text)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    reply_text = "Rate limit reached — please wait 15–60 seconds and try again. (Free tier: 15 requests/min)"
                    ph.warning(reply_text)
                else:
                    reply_text = f"Error: {e}"
                    ph.error(reply_text)

        st.session_state.messages.append({"role": "model", "parts": [reply_text]})

    # Footer controls
    if st.session_state.messages:
        col_a, col_b = st.columns([1, 5])
        with col_a:
            if st.button("🗑 Clear chat"):
                st.session_state.messages = []
                st.rerun()
        with col_b:
            transcript = "\n\n".join(
                f"**{'You' if m['role']=='user' else 'Assistant'}:** {m['parts'][0]}"
                for m in st.session_state.messages
            )
            st.download_button("⬇ Export transcript", data=transcript,
                               file_name="hematology_chat.md", mime="text/markdown")
