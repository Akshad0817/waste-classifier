import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model and class mapping once at startup
@st.cache_resource
def load_assets():
    model = load_model('waste_classifier.keras')
    with open('class_indices.json') as f:
        class_indices = json.load(f)
    # invert dict: {0: 'cardboard', 1: 'glass', ...}
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_assets()

# Map raw classes to disposal guidance (this is your product layer, not the model)
DISPOSAL_GUIDANCE = {
    'cardboard': 'Recyclable — flatten and place in the recycling bin.',
    'glass': 'Recyclable — rinse before placing in the recycling bin.',
    'metal': 'Recyclable — rinse before placing in the recycling bin.',
    'paper': 'Recyclable — keep dry and place in the recycling bin.',
    'plastic': 'Recyclable — check local rules for plastic type; rinse first.',
    'trash': 'Non-recyclable — dispose in general/landfill waste.',
}

st.set_page_config(page_title="Waste Classifier", page_icon="♻️")
st.title("♻️ Waste Classifier")
st.write("Upload a photo of a waste item and get an instant disposal recommendation.")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Preprocess to match training pipeline exactly
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    with st.spinner("Classifying..."):
        preds = model.predict(img_array)[0]
        pred_idx = np.argmax(preds)
        pred_class = idx_to_class[pred_idx]
        confidence = preds[pred_idx] * 100

    CONFIDENCE_THRESHOLD = 50.0  # below this, treat as uncertain / possibly not a recognized waste item

    if confidence < CONFIDENCE_THRESHOLD:
        st.subheader("Prediction: Uncertain")
        st.write(f"Top guess: {pred_class} ({confidence:.1f}% confidence)")
        st.warning(
            "This image doesn't clearly match a recognized waste category. "
            "It may not be a waste item, or it may be a type/condition (e.g. broken glass) "
            "the model wasn't trained to recognize well. Treat this result with caution."
        )
    else:
        st.subheader(f"Prediction: **{pred_class.upper()}**")
        st.write(f"Confidence: {confidence:.1f}%")
        st.info(DISPOSAL_GUIDANCE[pred_class])

    with st.expander("See all class probabilities"):
        for idx, prob in enumerate(preds):
            st.write(f"{idx_to_class[idx]}: {prob*100:.1f}%")

st.caption("Model: MobileNetV2 fine-tuned on TrashNet · ~77% validation accuracy · See README for known limitations")
