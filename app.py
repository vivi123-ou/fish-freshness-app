"""
🐟 Fish Eye Freshness Detection — VGG19
Chạy: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, gdown

# ── SỬA DÒNG NÀY: dán FILE_ID Google Drive của vgg19_best.pth ──
GDRIVE_FILE_ID = "PASTE_YOUR_FILE_ID_HERE"
# ────────────────────────────────────────────────────────────────

MODEL_PATH  = "vgg19_best.pth"
CLASS_NAMES = ["highly_fresh", "fresh", "not_fresh"]
CLASS_VI    = {"highly_fresh": "🟢 Rất Tươi", "fresh": "🟡 Tươi", "not_fresh": "🔴 Không Tươi"}
CLASS_CSS   = {"highly_fresh": "#d4edda", "fresh": "#fff3cd", "not_fresh": "#f8d7da"}
CLASS_FONT  = {"highly_fresh": "#155724", "fresh": "#856404", "not_fresh": "#721c24"}

st.set_page_config(page_title="Nhận Diện Độ Tươi Mắt Cá", page_icon="🐟")

# ── Tải model (chỉ 1 lần, cache lại) ──────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Đang tải model lần đầu (~550MB), chờ 1-2 phút..."):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                MODEL_PATH, quiet=False
            )
    net = models.vgg19(weights=None)
    net.classifier[6] = nn.Linear(4096, 3)
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    return net

# ── Transform & predict ────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(net, img: Image.Image):
    t = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(net(t), dim=1).squeeze().numpy()
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], probs

# ── UI ─────────────────────────────────────────────────────────
st.title("🐟 Nhận Diện Độ Tươi Mắt Cá")
st.caption("Mô hình VGG19 · Transfer Learning · Dataset: FFE Mendeley")
st.markdown("---")

net = load_model()

uploaded = st.file_uploader("📤 Upload ảnh mắt cá (JPG / PNG)", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(img, caption="Ảnh đã upload", use_column_width=True)

    with col2:
        label, probs = predict(net, img)
        bg   = CLASS_CSS[label]
        font = CLASS_FONT[label]

        st.markdown(
            f"""<div style="background:{bg};color:{font};border-radius:12px;
            padding:24px;text-align:center;font-size:1.6rem;font-weight:bold;
            margin-bottom:16px;">{CLASS_VI[label]}</div>""",
            unsafe_allow_html=True
        )

        st.markdown("**Xác suất:**")
        icons = {"highly_fresh":"🟢","fresh":"🟡","not_fresh":"🔴"}
        for cls, prob in zip(CLASS_NAMES, probs):
            st.write(f"{icons[cls]} {cls.capitalize()}: **{prob*100:.1f}%**")
            st.progress(float(prob))

else:
    st.info("👆 Upload ảnh mắt cá để bắt đầu.")

st.markdown("---")
st.caption("ĐH Mở TP.HCM · NCKH Sinh Viên 2025")
