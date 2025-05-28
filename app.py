import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient
import torch

# Set page config
st.set_page_config(page_title="AI Radiology Report Generator", layout="centered")

# Load BLIP
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

# Load HF Inference Client (✅ Working model)
# client = InferenceClient("tiiuae/falcon-7b-instruct", token=st.secrets["HF_TOKEN"])
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=st.secrets["HF_TOKEN"])


# Generate report from caption
def generate_radiology_report(caption: str) -> str:
    prompt = f"""
You are a professional radiologist.

A chest X-ray has been described as: "{caption}"

Write a radiology report including:

**Findings**: Describe anatomical observations.

**Impression**: Provide diagnostic summary.
"""
    response = client.text_generation(prompt=prompt, max_new_tokens=400, temperature=0.7)
    return response.strip()

# UI
st.title("🩻 AI Radiology Report Generator")
st.caption("Upload a chest X-ray. This AI will generate a structured report.")

uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Chest X-ray", use_column_width=True)

    with st.spinner("🔍 Describing image using BLIP..."):
        inputs = processor(image, return_tensors="pt")
        outputs = blip_model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        st.markdown(f"🧠 **Visual Description:** _{caption}_")

    with st.spinner("📝 Generating Report using Falcon-7B..."):
        report = generate_radiology_report(caption)
        st.markdown("### 📄 Generated Radiology Report")
        st.write(report)

    st.caption("⚠️ Educational prototype. Not for clinical use.")
