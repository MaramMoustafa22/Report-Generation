import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient
import torch

# Page config
st.set_page_config(page_title="AI Radiology Report Generator", layout="centered")

# Load BLIP model
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

# Setup Hugging Face Inference Client (Lightweight + Supported model)
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta", 
    token=st.secrets["HF_TOKEN"]
)

def generate_radiology_report(caption: str) -> str:
    prompt = f"""
You are a professional radiologist.

A chest X-ray image was described visually as follows:
"{caption}"

Write a detailed radiology report with two sections:

**Findings**: Describe the visual anatomical observations.

**Impression**: Provide a diagnostic summary or conclusion.
"""
    response = client.text_generation(prompt=prompt, max_new_tokens=400, temperature=0.7)
    return response.strip()

# UI
st.title("🩻 AI Radiology Report Generator")
st.caption("Upload a chest X-ray. The AI will describe the image and generate a formal radiology report.")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded X-ray", use_container_width=True)

    with st.spinner("🔍 Describing image with BLIP..."):
        inputs = processor(image, return_tensors="pt")
        outputs = blip_model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        st.markdown(f"🧠 **Visual Description:** _{caption}_")

    with st.spinner("📝 Generating Report..."):
        report = generate_radiology_report(caption)
        st.markdown("### 📄 Generated Radiology Report")
        st.write(report)

    st.caption("⚠️ This is a demo using pretrained models. Outputs are for research and educational use only.")
