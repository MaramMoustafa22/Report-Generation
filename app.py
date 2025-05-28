import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch

# Set page config
st.set_page_config(page_title="AI Radiology Report Generator", layout="centered")

# Load BLIP model for image captioning
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip()

# Load local text generation model using pipeline
@st.cache_resource
def load_text_pipeline():
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")

text_generator = load_text_pipeline()

# Function to generate report
def generate_radiology_report(caption: str) -> str:
    prompt = f"""
    You are a radiologist.

    Describe this chest X-ray based on visual description: "{caption}"

    Write a detailed radiology report with:
    **Findings**: Describe anatomical observations.
    **Impression**: Provide clinical interpretation.
    """
    result = text_generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]

# UI
st.title("🩻 AI Radiology Report Generator")
st.caption("Upload a chest X-ray to generate a professional-style report.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Captioning image..."):
        inputs = processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.markdown(f"🧠 **Visual Description:** _{caption}_")

    with st.spinner("📝 Generating report..."):
        report = generate_radiology_report(caption)
        st.markdown("### 📄 Generated Radiology Report")
        st.write(report)

    st.caption("⚠️ This is a prototype. Generated reports are not clinically validated.")
