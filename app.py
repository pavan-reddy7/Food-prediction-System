import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown
# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
GOOGLE_DRIVE_FILE_ID = "1lbC18PpYnTalr2ltTPYx9kvAor-pKxeD"
MODEL_PATH = "best_resnet50_food_gpu_v2_resumed.pth"
CONFIDENCE_THRESHOLD = 50.0

st.set_page_config(
    page_title="Indian Food Recognition",
    page_icon="🍛",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
# CSS  –  Premium dark "Netflix × AI Dashboard"
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset / base ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #0B0D0F !important;
    color: #F5F5F5 !important;
}

/* remove default top padding Streamlit adds */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 780px !important;
}

/* hide chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── HERO ─────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0B0D0F 0%, #15181C 100%);
    border-bottom: 1px solid #292D33;
    padding: 28px 32px 22px;
    margin: 0 -1rem 28px;
    text-align: center;
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,159,67,0.12);
    border: 1px solid rgba(255,159,67,0.35);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #FF9F43;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #F5F5F5;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.hero-sub {
    font-size: 0.9rem;
    color: #9CA3AF;
    margin: 0 0 16px;
}
.hero-badges {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}
.badge {
    background: #15181C;
    border: 1px solid #292D33;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: #9CA3AF;
    letter-spacing: 0.5px;
}

/* ── UPLOAD CARD ──────────────────────────────── */
.upload-card {
    background: #15181C;
    border: 1.5px dashed rgba(255,159,67,0.4);
    border-radius: 16px;
    padding: 36px 28px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 0 30px rgba(255,159,67,0.06);
    transition: border-color 0.3s, box-shadow 0.3s;
    cursor: pointer;
}
.upload-card:hover {
    border-color: rgba(255,159,67,0.75);
    box-shadow: 0 0 40px rgba(255,159,67,0.14);
}
.upload-icon {
    width: 52px; height: 52px;
    background: rgba(255,159,67,0.1);
    border: 1.5px solid rgba(255,159,67,0.3);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 14px;
    font-size: 1.4rem;
}
.upload-title {
    font-size: 1rem;
    font-weight: 700;
    color: #F5F5F5;
    margin-bottom: 5px;
}
.upload-sub {
    font-size: 0.82rem;
    color: #9CA3AF;
    margin-bottom: 18px;
}
.upload-formats {
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    color: #9CA3AF;
    text-transform: uppercase;
    margin-top: 12px;
}

/* hide Streamlit's own uploader chrome & expose it inside our card */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploadDropzone"] {
    background: rgba(255,159,67,0.08) !important;
    border: 1px solid rgba(255,159,67,0.3) !important;
    border-radius: 10px !important;
    color: #FF9F43 !important;
}

/* ── RESULT LAYOUT ────────────────────────────── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 24px 0;
    animation: fadeUp 0.5s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── IMAGE PANEL ──────────────────────────────── */
.img-panel {
    background: #15181C;
    border: 1px solid #292D33;
    border-radius: 16px;
    overflow: hidden;
}
.img-panel img {
    width: 100%;
    height: 260px;
    object-fit: cover;
    display: block;
    transition: transform 0.4s ease;
}
.img-panel:hover img { transform: scale(1.04); }

/* ── AI ANALYSIS CARD ─────────────────────────── */
.analysis-card {
    background: #15181C;
    border: 1px solid #292D33;
    border-radius: 16px;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    animation: fadeUp 0.55s ease;
    box-shadow: 0 0 40px rgba(255,159,67,0.07);
}
.ai-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #FF9F43;
    font-weight: 700;
}
.food-name-large {
    font-size: 1.65rem;
    font-weight: 800;
    color: #F5F5F5;
    line-height: 1.15;
    margin: 2px 0 0;
}
.conf-bar-outer {
    background: #1C2025;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin-top: 4px;
}
.conf-bar-inner {
    height: 8px;
    border-radius: 6px;
    background: linear-gradient(90deg, #FF9F43, #e67e22);
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
}
.conf-pct {
    font-size: 1.1rem;
    font-weight: 700;
    color: #FF9F43;
    margin-top: 2px;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(46,204,113,0.1);
    border: 1px solid rgba(46,204,113,0.3);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #2ECC71;
    margin-top: 4px;
    width: fit-content;
}

/* ── WARNING BOX ──────────────────────────────── */
.warn-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.35);
    border-radius: 14px;
    padding: 20px 22px;
    animation: fadeUp 0.5s ease;
}
.warn-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #FBBF24;
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 8px;
}
.warn-body {
    font-size: 0.85rem;
    color: #9CA3AF;
    line-height: 1.6;
}

/* ── FOOD INFO SECTION ────────────────────────── */
.food-section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #F5F5F5;
    margin: 28px 0 14px;
}
.info-cards-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 12px;
}
.info-card {
    background: #15181C;
    border: 1px solid #292D33;
    border-radius: 12px;
    padding: 18px;
}
.info-card-full {
    background: #15181C;
    border: 1px solid #292D33;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
}
.info-card-label {
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #FF9F43;
    font-weight: 700;
    margin-bottom: 8px;
}
.info-card-value {
    font-size: 0.9rem;
    color: #F5F5F5;
    line-height: 1.55;
}

/* ── TOP 5 EXPANDER ───────────────────────────── */
[data-testid="stExpander"] {
    background: #15181C !important;
    border: 1px solid #292D33 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #9CA3AF !important;
    padding: 14px 18px !important;
}
[data-testid="stExpander"] summary:hover { color: #F5F5F5 !important; }

.top5-label {
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #9CA3AF;
    font-weight: 700;
    margin-bottom: 14px;
}
.top5-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}
.top5-rank {
    font-size: 0.75rem;
    font-weight: 700;
    color: #9CA3AF;
    width: 26px;
    text-align: center;
    flex-shrink: 0;
}
.top5-name {
    font-size: 0.88rem;
    font-weight: 500;
    color: #F5F5F5;
    flex: 1;
}
.top5-pct {
    font-size: 0.82rem;
    font-weight: 600;
    color: #FF9F43;
    width: 50px;
    text-align: right;
    flex-shrink: 0;
}
.top5-bar-outer {
    width: 90px;
    background: #1C2025;
    border-radius: 4px;
    height: 5px;
    flex-shrink: 0;
}
.top5-bar-inner {
    height: 5px;
    border-radius: 4px;
    background: linear-gradient(90deg, #FF9F43, #e67e22);
}

/* ── MODEL INFO EXPANDER ──────────────────────── */
.model-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.model-table tr { border-bottom: 1px solid #1C2025; }
.model-table tr:last-child { border-bottom: none; }
.model-table td { padding: 10px 4px; }
.model-table td:first-child {
    color: #9CA3AF;
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    width: 40%;
    font-weight: 600;
}
.model-table td:last-child { color: #F5F5F5; font-weight: 500; }

.pipeline {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
}
.pipe-step {
    background: #1C2025;
    border: 1px solid #292D33;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.72rem;
    letter-spacing: 0.8px;
    color: #F5F5F5;
    font-weight: 600;
}
.pipe-arrow { color: #FF9F43; font-size: 0.9rem; }

/* ── ANALYZE ANOTHER BUTTON ───────────────────── */
.stButton > button {
    background: linear-gradient(90deg, #FF9F43, #e67e22) !important;
    color: #0B0D0F !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    letter-spacing: 0.4px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── FOOTER ───────────────────────────────────── */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #292D33;
    padding: 24px 0 8px;
    border-top: 1px solid #15181C;
    margin-top: 32px;
}

/* ── PROGRESS (built-in) override ────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #FF9F43, #e67e22) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOD INFORMATION
# ─────────────────────────────────────────────

FOOD_INFO = {
    "biryani":       {"category": "Rice Dish",      "ingredients": "Rice · Spices · Onion · Yogurt · Chicken",         "description": "Aromatic layered rice dish slow-cooked with whole spices, caramelized onions, and marinated meat or vegetables."},
    "naan":          {"category": "Indian Bread",   "ingredients": "Refined Flour · Yogurt · Yeast · Butter",          "description": "Soft leavened flatbread baked in a tandoor, often brushed with garlic butter."},
    "jalebi":        {"category": "Sweet",          "ingredients": "Flour · Sugar Syrup · Saffron · Oil",              "description": "Crispy spiral-shaped sweet soaked in fragrant saffron sugar syrup."},
    "lassi":         {"category": "Beverage",       "ingredients": "Yogurt · Milk · Rose Water · Sugar",               "description": "Cooling yogurt-based drink, sweet or salty, popular across North India."},
    "gulab_jamun":   {"category": "Sweet",          "ingredients": "Milk Solids · Flour · Cardamom · Sugar Syrup",     "description": "Soft milk-solid dumplings deep-fried and steeped in rose-cardamom syrup."},
    "modak":         {"category": "Sweet",          "ingredients": "Rice Flour · Coconut · Jaggery · Cardamom",        "description": "Steamed sweet dumplings with a coconut-jaggery filling, sacred to Lord Ganesha."},
    "poha":          {"category": "Breakfast",      "ingredients": "Flattened Rice · Onion · Mustard Seeds · Turmeric","description": "Light, flavorful breakfast dish made from flattened rice tempered with spices."},
    "dal_tadka":     {"category": "Main Course",    "ingredients": "Lentils · Ghee · Cumin · Garlic · Tomato",         "description": "Cooked yellow lentils finished with a smoky ghee and spice tempering."},
    "palak_paneer":  {"category": "Main Course",    "ingredients": "Spinach · Paneer · Garlic · Ginger · Cream",       "description": "Velvety spinach gravy with fresh cottage cheese cubes, a North Indian staple."},
    "butter_chicken":{"category": "Main Course",    "ingredients": "Chicken · Tomato · Butter · Cream · Fenugreek",    "description": "Tender chicken in a rich, mildly spiced tomato-butter cream sauce."},
    "chapati":       {"category": "Indian Bread",   "ingredients": "Whole Wheat Flour · Water · Salt",                 "description": "Thin unleavened whole-wheat flatbread cooked on a hot tawa."},
    "rasgulla":      {"category": "Sweet",          "ingredients": "Chhena · Semolina · Sugar · Rose Water",           "description": "Soft, spongy cottage-cheese balls simmered in light sugar syrup."},
    "mysore_pak":    {"category": "Sweet",          "ingredients": "Gram Flour · Ghee · Sugar",                        "description": "Crumbly, melt-in-mouth South Indian fudge made with roasted chickpea flour."},
    "adhirasam":     {"category": "Sweet",          "ingredients": "Rice Flour · Jaggery · Cardamom · Oil",            "description": "Traditional Tamil deep-fried sweet with a crisp exterior and chewy center."},
    "pootharekulu":  {"category": "Sweet",          "ingredients": "Rice Starch Sheets · Jaggery · Ghee · Sesame",     "description": "Delicate paper-thin Andhra sweet layered with jaggery and dry fruits."},
}

def download_model_from_drive():
    """Download model from Google Drive if not already present locally"""
    if os.path.exists(MODEL_PATH):
        return True

    try:
        st.info("📥 Downloading model from Google Drive (first time only)...")
        file_url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(file_url, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Model download failed: {e}")
        return False
# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not download_model_from_drive():
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(classes))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, classes, device


# ─────────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
# HERO — flush to top, no extra spacing
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1 class="hero-title">Indian Food Recognition</h1>
    <p class="hero-sub">Upload a photo of Indian food and let identify it instantly</p>
    <div class="hero-badges">
        <span class="badge">ResNet-50</span>
        <span class="badge">80 Food Classes</span>
        <span class="badge">Transfer Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UPLOAD SECTION
# ─────────────────────────────────────────────

# Session state key – incrementing it forces a fresh uploader widget
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible",
    key=f"uploader_{st.session_state.uploader_key}"
)

st.markdown(
    '<p class="upload-formats">JPG &nbsp;·&nbsp; PNG &nbsp;·&nbsp; WEBP &nbsp;·&nbsp; JPEG</p>',
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analysing image…"):
        model, classes, device = load_model()
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out   = model(tensor)
            probs = torch.softmax(out, dim=1)
            top5_probs, top5_idx = torch.topk(probs, 5)

    pred_idx     = top5_idx[0][0].item()
    pred_food    = classes[pred_idx]
    confidence   = top5_probs[0][0].item() * 100
    display_name = pred_food.replace("_", " ").title()

    # ── Two-column result layout ────────────────

    # Encode image to base64 for HTML embedding
    import io, base64
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    bar_width = int(min(confidence, 100))

    if confidence >= CONFIDENCE_THRESHOLD:
        analysis_html = f"""
        <div class="result-grid">
            <div class="img-panel">
                <img src="data:image/jpeg;base64,{b64}" alt="food image"/>
            </div>
            <div class="analysis-card">
                <div>
                    <div class="ai-label">AI Analysis</div>
                </div>
                <div>
                    <div class="ai-label" style="margin-bottom:4px;">Predicted Food</div>
                    <div class="food-name-large">{display_name}</div>
                </div>
                <div>
                    <div class="ai-label" style="margin-bottom:6px;">Model Confidence</div>
                    <div class="conf-bar-outer">
                        <div class="conf-bar-inner" style="width:{bar_width}%;"></div>
                    </div>
                    <div class="conf-pct">{confidence:.1f}%</div>
                </div>
                <div class="status-pill">✓ &nbsp;Analysis Complete</div>
            </div>
        </div>
        """
    else:
        analysis_html = f"""
        <div class="result-grid">
            <div class="img-panel">
                <img src="data:image/jpeg;base64,{b64}" alt="food image"/>
            </div>
            <div class="analysis-card">
                <div>
                    <div class="ai-label">AI Analysis</div>
                </div>
                <div>
                    <div class="ai-label" style="margin-bottom:4px;">Predicted Food</div>
                    <div class="food-name-large">{display_name}</div>
                </div>
                <div>
                    <div class="ai-label" style="margin-bottom:6px;">Model Confidence</div>
                    <div class="conf-bar-outer">
                        <div class="conf-bar-inner"
                             style="width:{bar_width}%;
                                    background:linear-gradient(90deg,#FBBF24,#f59e0b);">
                        </div>
                    </div>
                    <div class="conf-pct" style="color:#FBBF24;">{confidence:.1f}%</div>
                </div>
                <div class="warn-box" style="margin-top:4px;padding:12px 16px;">
                    <div class="warn-title">⚠ Low Confidence Prediction</div>
                    <div class="warn-body">
                        The model is may uncertain about this image.<br>
                        Try a clearer, well-lit food photograph.
                    </div>
                </div>
            </div>
        </div>
        """

    st.markdown(analysis_html, unsafe_allow_html=True)

    # ── Food Information ────────────────────────

    info = FOOD_INFO.get(pred_food, {
        "category":    "Indian Food",
        "ingredients": "Varies by recipe",
        "description": "A delicious Indian dish. More info coming soon."
    })

    st.markdown(f'<div class="food-section-title">{display_name}</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-cards-row">
        <div class="info-card">
            <div class="info-card-label">Category</div>
            <div class="info-card-value">{info["category"]}</div>
        </div>
        <div class="info-card">
            <div class="info-card-label">Common Ingredients</div>
            <div class="info-card-value">{info["ingredients"]}</div>
        </div>
    </div>
    <div class="info-card-full">
        <div class="info-card-label">About</div>
        <div class="info-card-value">{info["description"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Detailed Top 5 (collapsed) ──────────────

    with st.expander("⌄  Show detailed predictions"):

        st.markdown('<div class="top5-label">Model Predictions</div>',
                    unsafe_allow_html=True)

        for i in range(5):
            fname  = classes[top5_idx[0][i].item()].replace("_", " ").title()
            pct    = top5_probs[0][i].item() * 100
            rank   = f"0{i+1}" if i < 9 else str(i+1)
            rank_color = "#FF9F43" if i == 0 else "#9CA3AF"
            bar_w  = int(min(pct, 100))
            st.markdown(f"""
            <div class="top5-row">
                <div class="top5-rank" style="color:{rank_color};">{rank}</div>
                <div class="top5-name">{fname}</div>
                <div class="top5-bar-outer">
                    <div class="top5-bar-inner" style="width:{bar_w}%;"></div>
                </div>
                <div class="top5-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Analyze Another ─────────────────────────

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  Analyze Another Image"):
        st.session_state.uploader_key += 1   # new key → fresh uploader widget
        st.rerun()

# ─────────────────────────────────────────────
# MODEL INFO (always visible at bottom, collapsed)
# ─────────────────────────────────────────────

with st.expander("⌄  About the AI Model"):
    st.markdown("""
    <table class="model-table">
        <tr><td>Model</td><td>ResNet-50</td></tr>
        <tr><td>Approach</td><td>Transfer Learning (Fine-tuned)</td></tr>
        <tr><td>Food Classes</td><td>80</td></tr>
        <tr><td>Dataset</td><td>~4,000 Images</td></tr>
        <tr><td>Validation Accuracy</td><td>66.75%</td></tr>
        <tr><td>Framework</td><td>PyTorch</td></tr>
    </table>

    <div class="pipeline">
        <span class="pipe-step">IMAGE</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">PREPROCESSING</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">RESNET-50</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">CLASSIFICATION</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">FOOD INFORMATION</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown(
    '<div class="footer">Indian Food Recognition &amp; Information System · '
    'Powered by ResNet-50 &amp; Machine Learning</div>',
    unsafe_allow_html=True
)
