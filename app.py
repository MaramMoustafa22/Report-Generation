import streamlit as st
st.set_page_config(page_title="AI Radiology Report Generator", layout="centered")

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient
import torch

# Load BLIP model
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

# Load LLM client (Mistral-7B)
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1")

# Generate a structured radiology report
def generate_radiology_report(caption: str) -> str:
    prompt = f"""
You are a professional radiologist.

A chest X-ray has been described visually as: "{caption}"

Write a formal radiology report including:

**Findings**: Describe anatomical details and radiological observations.

**Impression**: Provide clinical interpretation or diagnostic summary.
"""
    response = client.text_generation(prompt=prompt, max_new_tokens=400, temperature=0.7)
    return response.strip()

# Streamlit UI
st.title("AI Radiology Report Generator")
st.caption("Upload a chest X-ray. The AI will generate a structured radiology report based on the visual description.")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded X-ray", use_column_width=True)

    with st.spinner("🔍 Analyzing image with BLIP..."):
        inputs = processor(image, return_tensors="pt")
        outputs = blip_model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        st.markdown(f"🧠 **Visual Description:** _{caption}_")

    with st.spinner("📝 Generating Radiology Report..."):
        report = generate_radiology_report(caption)
        st.markdown("### 📄 Generated Radiology Report")
        st.write(report)

    st.markdown("---")
    st.caption("⚠️ This is a prototype for educational purposes. Outputs are not medically validated.")
    