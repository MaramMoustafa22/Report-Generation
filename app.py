import os
import asyncio

# --- Fix: Prevent Streamlit + torch watcher crash ---
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# --- Fix: Patch asyncio event loop error ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient

# Load BLIP model and processor for image captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load Hugging Face Inference Client for text generation
@st.cache_resource
def load_hf_client():
    hf_token = st.secrets["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"]
    return InferenceClient(token=hf_token)

# Generate radiology report using Falcon-7B
def generate_radiology_report(caption):
    client = load_hf_client()
    prompt = f"""You are a radiology assistant AI. Generate a professional radiology report based on the provided image description.

**Findings**: {caption}

**Impression**: Provide diagnostic summary."""
    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=400,
        temperature=0.7,
        model="tiiuae/falcon-7b-instruct"  # Explicit model
    )
    return response.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Radiology Report Generator", layout="centered")
st.title("📄 Radiology Report Generator")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Generating image caption..."):
        processor, blip_model = load_blip_model()
        inputs = processor(images=image, return_tensors="pt")
        outputs = blip_model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        st.markdown(f"🧠 **Visual Description:** _{caption}_")

    with st.spinner("📝 Generating Report using Falcon-7B..."):
        try:
            report = generate_radiology_report(caption)
            st.markdown("### 📄 Generated Radiology Report")
            st.write(report)
        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")
