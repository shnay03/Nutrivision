import streamlit as st
import numpy as np
import cv2
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="NutriVision – Stunting Detection",
    layout="wide"
)

st.markdown("""
<style>
            
/* ================= HEADER / TOP BAR ================= */
header[data-testid="stHeader"] {
    background-color: #16263A;
}

/* Hilangkan garis bawah header */
header[data-testid="stHeader"]::after {
    background: none;
}

/* ================= ROOT APP ================= */
[data-testid="stAppViewContainer"] {
    background-color: #16263A;
}

/* ================= KONTEN UTAMA ================= */
[data-testid="stAppViewContainer"] .block-container {
    background-color: #16263A;
    padding-top: 3rem;
}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {
    background-color: #1E3550;
}

/* ================= TEKS ================= */
h1, h2, h3, h4, h5, h6, p, label, span {
    color: #E6EEF7 !important;
}

/* ================= UPLOAD BOX ================= */
section[data-testid="stFileUploader"] {
    background-color: rgba(36, 56, 79, 0.75);
    border-radius: 12px;
    padding: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(2px);
}

/* ================= BUTTON ================= */
button {
    background-color: #3A7BD5 !important;
    color: white !important;
    border-radius: 8px;
}

/* ================= SELECTBOX (DROPDOWN) ================= */
div[data-testid="stSelectbox"] > div {
    background-color: rgba(20, 35, 55, 0.85);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.12);
}

/* teks di dalam selectbox */
div[data-testid="stSelectbox"] span {
    color: #E6EEF7 !important;
    font-weight: 500;
}

/* dropdown arrow */
div[data-testid="stSelectbox"] svg {
    fill: #AFC6E9;
}

/* hover */
div[data-testid="stSelectbox"] > div:hover {
    background-color: rgba(30, 50, 75, 0.9);
}

            
</style>
""", unsafe_allow_html=True)



# ================= LOAD MODELS =================
cnn_model = load_model("best_model.h5")

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

faster_model = fasterrcnn_resnet50_fpn(weights=None)
in_features = faster_model.roi_heads.box_predictor.cls_score.in_features
faster_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
faster_model.load_state_dict(torch.load("best_frcnn_antro.pth", map_location=device), strict=False)
faster_model.to(device)
faster_model.eval()

# ================= MEDIAPIPE =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# ================= ANTHRO FEATURES =================
def calculate_3d_distance(p1, p2):
    return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)

def extract_anthropometric_features(landmarks):
    if landmarks is None:
        return np.zeros(3)
    try:
        L = [landmarks.landmark[123], landmarks.landmark[50], landmarks.landmark[36]]
        R = [landmarks.landmark[352], landmarks.landmark[323], landmarks.landmark[266]]
        cheek = (calculate_3d_distance(L[0],L[1]) + calculate_3d_distance(R[0],R[1]))/2

        eye = (abs(landmarks.landmark[33].y-landmarks.landmark[159].y) +
               abs(landmarks.landmark[362].y-landmarks.landmark[386].y))/2

        width = calculate_3d_distance(landmarks.landmark[454], landmarks.landmark[234])
        height = calculate_3d_distance(landmarks.landmark[10], landmarks.landmark[152])

        return np.array([cheek, eye, width/height if height!=0 else 0])
    except:
        return np.zeros(3)

# ================= MODEL RUN =================
def run_cnn(img_rgb):
    img = cv2.resize(img_rgb,(224,224))/255.0
    res = face_mesh.process(img_rgb)
    lm = res.multi_face_landmarks[0] if res.multi_face_landmarks else None
    feat = extract_anthropometric_features(lm)

    pred = cnn_model.predict({
        "image": np.expand_dims(img,0),
        "anthropo": np.expand_dims(feat,0)
    })[0][0]

    return ("STUNTING" if pred>0.5 else "NORMAL"), pred, feat, lm

# ======================================================
# FASTER R-CNN 
# ======================================================

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class BoxHeadWithAntro(nn.Module):
    def __init__(self, original, antro_dim):
        super().__init__()
        self.fc1 = original.fc6
        self.fc2 = original.fc7
        self.antro_fc = nn.Linear(antro_dim, 256)
        self.fusion = nn.Linear(1024+256, 1024)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        antro = self.antro_feats
        repeat = max(x.size(0)//antro.size(0),1)
        antro = antro.repeat_interleave(repeat,0)[:x.size(0)]
        antro = F.relu(self.antro_fc(antro))

        x = torch.cat([x, antro], 1)
        return F.relu(self.fusion(x))


class FasterRCNNWithAntro(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, targets=None, feats=None):
        self.model.roi_heads.box_head.antro_feats = torch.stack(feats)
        return self.model(images, targets)


# ======================================================
# LOAD FASTER R-CNN
# ======================================================

anchor_generator = AnchorGenerator(
    sizes=((16,), (32,), (64,), (128,), (256,)),
    aspect_ratios=((0.8,1.0,1.2),)*5
)

roi_pooler = MultiScaleRoIAlign(
    featmap_names=["0","1","2","3"],
    output_size=7,
    sampling_ratio=2
)

base = fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

base.roi_heads.box_predictor.cls_score = nn.Linear(1024,3)
base.roi_heads.box_predictor.bbox_pred = nn.Linear(1024,3*4)
base.roi_heads.box_head = BoxHeadWithAntro(base.roi_heads.box_head, 3)

faster_model = FasterRCNNWithAntro(base).to(device)

# LOAD TANPA strict=False
faster_model.load_state_dict(
    torch.load("best_frcnn_antro.pth", map_location=device)
)

faster_model.eval()


def run_faster(img_rgb):
    img_tensor = transform(img_rgb).to(device)

    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        lm = None
        feat = torch.zeros(3, dtype=torch.float32).to(device)
        feat_np = np.zeros(3)
    else:
        lm = res.multi_face_landmarks[0]
        feat_np = extract_anthropometric_features(lm)
        feat = torch.tensor(feat_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = faster_model(
            [img_tensor],
            feats=[feat]
        )[0]

    if len(output["scores"]) == 0:
        prob = 0.0
    else:
        idx = output["scores"].argmax()
        prob = output["scores"][idx].item()

    return output, prob, lm, feat_np



# ================= DRAW =================
def draw_visual(img_bgr, lm, preds, cnn_label, features, prob=None, model_choice=None):
    h, w, _ = img_bgr.shape

    # # ================= MODEL LABEL =================
    # if model_choice is not None:
    #     cv2.putText(
    #         img_bgr,
    #         f"Model: {model_choice}",
    #         (10, 25),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.8,
    #         (255, 255, 0),
    #         2
    #     )

    # ================= FACEMESH =================
    if lm is not None:
        for p in lm.landmark:
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(img_bgr, (x, y), 1, (0, 255, 0), -1)

    # ==========================================================
    # ===================== CNN MODE ===========================
    # ==========================================================
    if model_choice == "CNN":

        if prob is not None:

            status = "STUNTING" if prob > 0.4 else "NORMAL"

            cv2.putText(
                img_bgr,
                f"Status: {status}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if status == "STUNTING" else (0, 255, 0),
                2
            )

            cv2.putText(
                img_bgr,
                f"Confidence: {prob:.2f}",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    # ==========================================================
    # ================= FASTER R-CNN MODE ======================
    # ==========================================================
    if model_choice == "Faster R-CNN":

        if preds is not None and len(preds.get("scores", [])) > 0:

            # Ambil index dengan score tertinggi
            idx = preds["scores"].argmax()

            box = preds["boxes"][idx]
            score = preds["scores"][idx]
            label = preds["labels"][idx]

            x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)

            # Training: 1 = Normal, 2 = Stunting
            if label.item() == 2:
                cls = "STUNTING"
                color = (0, 0, 255)  # merah
            else:
                cls = "NORMAL"
                color = (0, 255, 0)  # hijau

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)

            cv2.putText(
                img_bgr,
                f"{cls} ({score:.2f})",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    return img_bgr

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    model_choice = st.selectbox(
        "Model Prediksi",
        ["CNN", "Faster R-CNN"]
    )
    mode = st.radio("Sumber Citra", ["Upload Gambar", "Kamera"])

# ================= MAIN =================
st.markdown("# NutriVision")
st.markdown("### Sistem Deteksi Dini Stunting Berbasis Citra Wajah")

col1, col2 = st.columns([1,2])

with col1:
    st.markdown("Input Citra")
    img_bgr = None

    if mode=="Upload Gambar":
        file = st.file_uploader("Unggah Foto Anak", ["jpg","png","jpeg"])
        if file:
            img_bgr = cv2.imdecode(np.frombuffer(file.read(),np.uint8),cv2.IMREAD_COLOR)

    else:
        cam = st.camera_input("Ambil Foto")
        if cam:
            img_bgr = cv2.imdecode(
                np.frombuffer(cam.getvalue(), np.uint8),
                cv2.IMREAD_COLOR
            )
            
            img_bgr = cv2.convertScaleAbs(img_bgr, alpha=1.1, beta=10)

            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    

if img_bgr is not None:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Reset dulu supaya tidak ada nilai sisa
    preds = None
    prob = None
    feat = None
    lm = None

    if model_choice == "CNN":

        label, prob, feat, lm = run_cnn(img_rgb)

    elif model_choice == "Faster R-CNN":

        preds, prob, lm, feat = run_faster(img_rgb)

        if preds is None or len(preds.get("scores", [])) == 0:
            label = "TIDAK TERDETEKSI"
            prob = None  # Faster tidak tampilkan CNN prob
        else:
            idx = preds["scores"].argmax()
            det_label = preds["labels"][idx].item()
            label = "STUNTING" if det_label == 2 else "NORMAL"
      

    st.markdown("### Hasil Deteksi")

    if label == "STUNTING":
        st.markdown("**Status:** Stunting")
    elif label == "NORMAL":
        st.markdown("**Status:** Normal")

    st.divider()

    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown("#### Visualisasi Citra")
        # vis = draw_result(img_bgr.copy(), lm, preds, label, feat)
        vis = draw_visual(
            img_bgr.copy(),
            lm,
            preds,
            label,
            feat,
            prob,
            model_choice
        )

        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        if feat is not None:
            st.markdown("#### Fitur Geometri Wajah")
            st.table({
                "Parameter": ["Cheek Depth", "Eye Depth", "Face Ratio"],
                "Nilai": [
                    f"{feat[0]:.4f}",
                    f"{feat[1]:.4f}",
                    f"{feat[2]:.4f}"
                ]
            })
