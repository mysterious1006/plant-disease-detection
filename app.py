import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "artifacts/plant_disease_model.keras"
LABELS_PATH = "artifacts/labels.json"

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
st.title("Plant Disease Detection")
st.caption("MobileNetV2 deep learning prototype")

st.warning(
    "Educational prototype. Predictions may be incorrect; "
    "verify important crop-health decisions with an agricultural expert."
)

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return model, labels

try:
    model, labels = load_model_and_labels()
except Exception as e:
    st.error(f"Model files not found. Copy trained artifacts first. Details: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Upload a leaf photograph",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if st.button("Analyze leaf", type="primary"):
        resized = image.resize((224, 224))
        x = np.asarray(resized, dtype=np.float32)[None, ...]

        scores = model.predict(x, verbose=0)[0]
        best = int(np.argmax(scores))

        st.subheader("Prediction")
        st.write(labels[best].replace("___", " — ").replace("_", " "))
        st.metric("Model score", f"{scores[best]:.1%}")

        st.write("Top predictions")
        for idx in np.argsort(scores)[::-1][:5]:
            st.write(
                f"{labels[int(idx)].replace('___', ' — ').replace('_', ' ')}: "
                f"{scores[idx]:.1%}"
            )

        st.caption(
            "Model scores are not necessarily calibrated probabilities. "
            "A high score does not guarantee a correct diagnosis."
        )
