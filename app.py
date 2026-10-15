
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================================================
# CONFIGURATION — CPU ONLY
# =========================================================
APP_DIR = Path(__file__).resolve().parent


def find_asset(filename, alternatives=()):
    """Find an asset beside app.py or in the bundled artifacts directory."""
    for directory in (APP_DIR, APP_DIR / "artifacts"):
        for candidate_name in (filename, *alternatives):
            candidate = directory / candidate_name
            if candidate.is_file():
                return candidate
    # Keep the expected path for a useful missing-file message.
    return APP_DIR / filename


MODEL_PATH = find_asset("best_model.keras", ("plant_disease_model.keras",))
LABELS_PATH = find_asset("class_names.json", ("labels.json",))
IMAGE_SIZE = (224, 224)

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

st.set_page_config(
    page_title="Verdant AI | Plant Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM DESIGN SYSTEM
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --forest: #173D2B;
        --forest2: #24583D;
        --sage: #DDE9D9;
        --mint: #EFF5EC;
        --cream: #F8F8F2;
        --ink: #203329;
        --muted: #718076;
        --line: #E3E9DF;
        --white: #FFFFFF;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%, #E5EFDF 0%, transparent 30%),
            linear-gradient(180deg, #F8F9F4 0%, #F4F6EF 100%);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(248,249,244,0.88);
    }

    [data-testid="stSidebar"] {
        background: #173D2B;
        border-right: 1px solid #315A43;
    }

    [data-testid="stSidebar"] * {
        color: #F3F7EF !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #D5E3D7 !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.7px;
        color: var(--forest) !important;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        padding: 0.3rem 0 1.5rem 0;
    }

    .brand span { color: #B6D9A7; }

    .eyebrow {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        color: #68816D;
        margin-bottom: 0.65rem;
    }

    .hero {
        background: linear-gradient(125deg, #173D2B 0%, #24583D 62%, #39724C 100%);
        border-radius: 25px;
        padding: 2.4rem 2.5rem;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 18px 45px rgba(27,66,43,0.13);
        margin-bottom: 1.4rem;
    }

    .hero::after {
        content: "❋";
        position: absolute;
        right: 3%;
        top: -55px;
        font-size: 240px;
        line-height: 1;
        color: rgba(255,255,255,0.055);
        transform: rotate(-20deg);
        pointer-events: none;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: clamp(2rem, 4vw, 3.1rem);
        font-weight: 800;
        line-height: 1.12;
        max-width: 700px;
        margin: 0.2rem 0 1rem 0;
    }

    .hero p {
        color: #D9E8DA;
        font-size: 1rem;
        line-height: 1.75;
        max-width: 680px;
        margin-bottom: 1.2rem;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(219,241,208,0.13);
        border: 1px solid rgba(220,241,214,0.22);
        border-radius: 30px;
        padding: 0.45rem 0.8rem;
        color: #E5F2DD;
        font-size: 0.76rem;
        font-weight: 600;
        margin: 0 0.35rem 0.35rem 0;
    }

    .metric-card {
        background: rgba(255,255,255,0.84);
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 1.15rem 1.2rem;
        min-height: 125px;
        box-shadow: 0 6px 20px rgba(28,61,38,0.035);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .metric-value {
        color: var(--forest);
        font-family: 'Manrope', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .metric-note {
        color: #7D8A7E;
        font-size: 0.72rem;
        margin-top: 0.25rem;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: var(--forest);
        font-weight: 800;
        font-size: 1.35rem;
        letter-spacing: -0.5px;
        margin: 1.4rem 0 0.35rem 0;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.65;
        margin-bottom: 1.1rem;
    }

    .panel {
        background: rgba(255,255,255,0.88);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.3rem;
        box-shadow: 0 8px 25px rgba(28,61,38,0.035);
        margin-bottom: 1rem;
    }

    .result-card {
        background: linear-gradient(145deg, #EAF3E5, #F7FAF3);
        border: 1px solid #D7E6D0;
        border-radius: 21px;
        padding: 1.5rem;
        margin: 0.5rem 0 1rem 0;
    }

    .result-kicker {
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #66816A;
        font-size: 0.72rem;
        font-weight: 800;
    }

    .result-name {
        color: #173D2B;
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.3;
        margin: 0.65rem 0;
        overflow-wrap: anywhere;
    }

    .result-copy {
        color: #617467;
        font-size: 0.85rem;
        line-height: 1.7;
    }

    .status-pill {
        display: inline-block;
        padding: 0.4rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        background: #DDEBDD;
        color: #24583D;
    }

    .status-pill.warning {
        background: #F8E9CE;
        color: #8B5B16;
    }

    .mini-label {
        color: #758278;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .mini-value {
        color: #234B32;
        font-size: 1.05rem;
        font-weight: 800;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
        border: 1px solid #24583D;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: #24583D;
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: #173D2B;
        border-color: #173D2B;
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] {
        background: #F9FBF7;
        border: 1px dashed #A9C2A4;
        border-radius: 16px;
        padding: 0.6rem;
    }

    [data-testid="stFileUploader"] section {
        border: none;
    }

    [data-testid="stProgressBar"] > div > div {
        background-color: #54855B;
    }

    .footer {
        color: #839087;
        text-align: center;
        font-size: 0.76rem;
        padding: 1.8rem 0 0.5rem 0;
        border-top: 1px solid #E1E8DE;
        margin-top: 2rem;
    }

    .small-muted {
        color: #77867A;
        font-size: 0.77rem;
        line-height: 1.7;
    }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.65);
        border: 1px solid var(--line);
        border-radius: 13px;
    }

    @media (max-width: 768px) {
        .hero { padding: 1.5rem; }
        .hero h1 { font-size: 2rem; }
        .block-container { padding-top: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


@st.cache_data(show_spinner=False)
def load_class_names():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)

    if isinstance(names, dict):
        names = [
            names[k]
            for k in sorted(names, key=lambda x: int(x))
        ]

    return names


def pretty_label(label):
    """Convert raw dataset labels into readable display names."""
    label = str(label)
    label = label.replace("___", " — ")
    label = label.replace("__", " ")
    label = label.replace("_", " ")
    label = label.replace("(", " (")
    label = " ".join(label.split())
    return label


def plant_name(label):
    """Extract plant name from a PlantVillage-style class label."""
    return str(label).split("___")[0].replace("_", " ").strip()


def health_status(label):
    """Detect the healthy category from its class label."""
    return "healthy" in str(label).lower()


def make_prediction(model, image):
    # The model already contains MobileNetV2 rescaling.
    # Feed raw RGB pixel values in the 0–255 range.
    resized = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    start = time.perf_counter()
    probabilities = model.predict(array, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    return probabilities, elapsed_ms


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        '<div class="brand">✳ VERDANT<span>.AI</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Plant intelligence")
    st.markdown(
        "A local AI assistant for exploring plant leaf health "
        "through computer vision."
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["🌱  Disease Scanner", "📊  Model Overview", "ℹ️  About Project"],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("**MODEL STATUS**")
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        st.success("Model files detected")
    else:
        st.warning("Model files need attention")

    st.markdown(
        """
        <div class="small-muted">
        Architecture: MobileNetV2<br>
        Input: 224 × 224 RGB<br>
        Classes: 38<br>
        Runtime: Local CPU
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        '<div class="small-muted">Designed for academic research and '
        'plant-health exploration.</div>',
        unsafe_allow_html=True,
    )

# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow" style="color:#B9D9B1;">
            COMPUTER VISION · AGRICULTURAL AI
        </div>
        <h1>Every leaf has a story.<br>Let AI help you read it.</h1>
        <p>
            Explore plant leaf health with a lightweight deep-learning
            model. Upload a leaf image to get a predicted class,
            confidence score, and the most likely alternatives —
            all processed locally on your laptop.
        </p>
        <span class="hero-tag">✳ MobileNetV2</span>
        <span class="hero-tag">38 classes</span>
        <span class="hero-tag">CPU inference</span>
        <span class="hero-tag">Local processing</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# METRIC CARDS
# =========================================================
metrics = [
    ("TEST ACCURACY", "95.74%", "Held-out test set"),
    ("MACRO F1-SCORE", "94.58%", "Equal weight per class"),
    ("MEDIAN LATENCY", "47.78 ms", "Supplied benchmark"),
    ("PLANT CLASSES", "38", "Healthy + disease classes"),
]

cols = st.columns(4)
for col, (label, value, note) in zip(cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# PAGE 1: DISEASE SCANNER
# =========================================================
if page == "🌱  Disease Scanner":

    st.markdown(
        '<div class="section-heading">Leaf health scanner</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Upload a clear, well-lit image '
        'of a single plant leaf for the best chance of a useful '
        'prediction.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.02, 0.98], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 01 / Upload your leaf")
        st.markdown(
            '<div class="small-muted">Supported formats: JPG, JPEG, PNG. '
            'For more reliable predictions, capture the leaf in focus '
            'with minimal obstruction.</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            key="leaf_upload",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(
                    image,
                    caption="Image ready for analysis",
                    use_container_width=True,
                )

                st.markdown(
                    f"""
                    <div style="display:flex;gap:20px;flex-wrap:wrap;
                    padding:0.7rem 0;">
                        <div>
                            <div class="mini-label">IMAGE SIZE</div>
                            <div class="mini-value">{image.width} × {image.height}</div>
                        </div>
                        <div>
                            <div class="mini-label">COLOR MODE</div>
                            <div class="mini-value">RGB</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                run_scan = st.button(
                    "✦  Analyze leaf",
                    type="primary",
                    use_container_width=True,
                )

            except Exception:
                st.error("This image could not be opened. Please try another.")
                image = None
                run_scan = False
        else:
            image = None
            run_scan = False
            st.info("Your uploaded image will appear here.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 02 / AI analysis")
        st.markdown(
            '<div class="small-muted">The model converts the image to '
            '224 × 224 RGB pixels and predicts one of 38 learned classes. '
            'The model runs on your laptop CPU.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background:#F0F5ED;border-radius:14px;
            padding:1rem;margin-top:1.1rem;">
                <div style="font-size:1.7rem;">🌿</div>
                <div style="font-weight:800;color:#24583D;
                margin:0.45rem 0;">Ready when you are</div>
                <div style="font-size:0.83rem;color:#728273;
                line-height:1.6;">
                    Upload an image, then select Analyze leaf to view
                    the model's prediction and confidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    if run_scan and image is not None:
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            st.error(
                "Model files were not found. Place best_model.keras and "
                "class_names.json beside app.py."
            )
        else:
            try:
                with st.spinner("Analyzing visual features..."):
                    model = load_model()
                    labels = load_class_names()
                    probabilities, elapsed_ms = make_prediction(model, image)

                if len(labels) != len(probabilities):
                    st.error(
                        f"Label mismatch: the model returned "
                        f"{len(probabilities)} outputs but "
                        f"{len(labels)} labels were loaded. "
                        "Check the label ordering and file."
                    )
                else:
                    top_indices = np.argsort(probabilities)[::-1][:3]
                    best_idx = int(top_indices[0])
                    best_raw = labels[best_idx]
                    best_name = pretty_label(best_raw)
                    confidence = float(probabilities[best_idx])
                    healthy = health_status(best_raw)

                    st.markdown(
                        '<div class="section-heading">Your analysis</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="section-subtitle">Prediction generated '
                        'by your trained MobileNetV2 model.</div>',
                        unsafe_allow_html=True,
                    )

                    res_col, detail_col = st.columns([1.1, 0.9], gap="large")

                    with res_col:
                        pill_class = "" if healthy else "warning"
                        pill_text = (
                            "HEALTHY CLASS PREDICTION"
                            if healthy
                            else "DISEASE CLASS PREDICTION"
                        )

                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-kicker">
                                    PRIMARY PREDICTION
                                </div>
                                <div class="result-name">{best_name}</div>
                                <span class="status-pill {pill_class}">
                                    {pill_text}
                                </span>
                                <div style="margin-top:1.3rem;
                                display:flex;justify-content:space-between;
                                align-items:center;">
                                    <span style="color:#667C69;
                                    font-size:0.84rem;font-weight:600;">
                                        Model confidence
                                    </span>
                                    <span style="color:#173D2B;
                                    font-size:1.3rem;font-weight:800;">
                                        {confidence * 100:.2f}%
                                    </span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.progress(
                            max(0.0, min(1.0, confidence)),
                            text="Top-class probability",
                        )

                        st.markdown(
                            f"""
                            <div style="display:flex;gap:25px;
                            flex-wrap:wrap;margin:0.8rem 0 1rem 0;">
                                <div>
                                    <div class="mini-label">PLANT</div>
                                    <div class="mini-value">
                                        {plant_name(best_raw)}
                                    </div>
                                </div>
                                <div>
                                    <div class="mini-label">INFERENCE TIME</div>
                                    <div class="mini-value">
                                        {elapsed_ms:.2f} ms
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with detail_col:
                        st.markdown(
                            '<div class="panel"><h4>Other likely matches</h4>',
                            unsafe_allow_html=True,
                        )

                        for rank, idx in enumerate(top_indices, start=1):
                            label = pretty_label(labels[int(idx)])
                            score = float(probabilities[int(idx)])

                            st.markdown(
                                f"""
                                <div style="display:flex;
                                justify-content:space-between;gap:10px;
                                margin-top:1rem;margin-bottom:0.25rem;">
                                    <span style="color:#304A38;
                                    font-size:0.82rem;font-weight:600;">
                                        {rank}. {label}
                                    </span>
                                    <span style="color:#426D4A;
                                    font-size:0.82rem;font-weight:800;">
                                        {score*100:.2f}%
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            st.progress(max(0.0, min(1.0, score)))

                        st.markdown("</div>", unsafe_allow_html=True)

                    with st.expander("View all 38 class probabilities"):
                        ranked = np.argsort(probabilities)[::-1]
                        for idx in ranked:
                            st.write(
                                f"**{pretty_label(labels[int(idx)])}** — "
                                f"{probabilities[int(idx)]*100:.3f}%"
                            )

                    st.caption(
                        "Confidence represents the model's output probability, "
                        "not a guarantee of correctness. Predictions may be less "
                        "reliable on field images unlike the training dataset."
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "Confirm that the model file, class-name file, "
                    "and preprocessing match the training workflow."
                )

    st.markdown(
        """
        <div class="footer">
            VERDANT.AI · Plant Disease Intelligence<br>
            Local inference · MobileNetV2 · Academic prototype
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# PAGE 2: MODEL OVERVIEW
# =========================================================
elif page == "📊  Model Overview":

    st.markdown(
        '<div class="section-heading">Inside the model</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">An overview of the reported '
        'evaluation results and model configuration.</div>',
        unsafe_allow_html=True,
    )

    a, b = st.columns(2, gap="large")

    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Evaluation summary")

        st.write("**Test accuracy**")
        st.progress(0.9574, text="95.74%")

        st.write("**Macro F1-score**")
        st.progress(0.9458, text="94.58%")

        st.write("**Weighted F1-score**")
        st.progress(0.96, text="Approximately 96.00%")

        st.write("**Test samples**")
        st.markdown("### 10,948")
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Architecture profile")

        rows = [
            ("Backbone", "MobileNetV2"),
            ("Input", "224 × 224 × 3"),
            ("Classes", "38"),
            ("Total parameters", "2,306,662"),
            ("Trainable parameters", "48,678"),
            ("Non-trainable parameters", "2,257,984"),
        ]

        for label, value in rows:
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;
                gap:12px;border-bottom:1px solid #E7ECE4;
                padding:0.75rem 0;">
                    <span style="color:#78857A;font-size:0.82rem;">
                        {label}
                    </span>
                    <span style="color:#234B32;font-weight:700;
                    text-align:right;font-size:0.83rem;">
                        {value}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Reported inference benchmark")

    x, y, z = st.columns(3)
    with x:
        st.metric("Median latency", "47.78 ms")
    with y:
        st.metric("95th percentile", "52.98 ms")
    with z:
        st.metric("Input batch size", "1")

    st.warning(
        "The supplied latency benchmark's hardware was not confirmed as "
        "your laptop. The actual latency shown on the scanner page is "
        "measured during inference on the current machine."
    )

    st.markdown("#### Methodology")

    steps = [
        ("01", "Dataset", "54,304 images across 38 classes"),
        ("02", "Preprocessing", "RGB conversion and 224 × 224 resizing"),
        ("03", "Training", "Transfer learning and fine-tuning"),
        ("04", "Evaluation", "Held-out test set and class-wise metrics"),
        ("05", "Deployment", "Local CPU inference"),
    ]

    for num, title, description in steps:
        st.markdown(
            f"""
            <div class="panel" style="display:flex;gap:18px;
            align-items:center;padding:1rem 1.2rem;">
                <div style="min-width:43px;height:43px;
                border-radius:13px;background:#E5EFE0;
                display:flex;align-items:center;justify-content:center;
                color:#24583D;font-weight:800;">{num}</div>
                <div>
                    <div style="font-weight:800;color:#234B32;">
                        {title}
                    </div>
                    <div class="small-muted">{description}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer">Reported metrics are from the supplied '
        'evaluation logs. They are not independently re-evaluated here.</div>',
        unsafe_allow_html=True,
    )

# =========================================================
# PAGE 3: ABOUT PROJECT
# =========================================================
else:
    st.markdown(
        '<div class="section-heading">About the project</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">A student deep-learning project '
        'exploring efficient plant-health classification.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
            <div class="eyebrow">PROJECT VISION</div>
            <h2 style="margin-top:0;">
                Making plant image analysis more accessible.
            </h2>
            <p style="color:#637568;line-height:1.85;">
                This project explores how a lightweight convolutional
                neural network can classify plant leaf images into
                healthy and disease-related categories. The goal is to
                create a practical prototype that runs on a regular
                laptop without depending on a remote inference server.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            """
            <div class="panel">
                <h4>Technology stack</h4>
                <p class="small-muted">
                    TensorFlow / Keras<br>
                    MobileNetV2<br>
                    Streamlit<br>
                    NumPy and Pillow<br>
                    TensorFlow Lite export workflow
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="panel">
                <h4>Responsible interpretation</h4>
                <p class="small-muted">
                    The model predicts one of its learned classes.
                    It does not replace an agronomist or laboratory
                    diagnosis. Real-world performance must be validated
                    on representative field images.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer">
            Built with TensorFlow, MobileNetV2, and Streamlit<br>
            Lightweight and Robust Deep Learning for Real-Time
            Plant Disease Detection on Edge Devices
        </div>
        """,
        unsafe_allow_html=True,
    )
