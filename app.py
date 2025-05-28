import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch

st.set_page_config(page_title="AI Radiology Report Generator", layout="centered")

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip()

@st.cache_resource
def load_text_pipeline():
    return pipeline("text-generation", model="sshleifer/distill-gpt2")  # ✅ lightweight

text_generator = load_text_pipeline()

def generate_radiology_report(caption: str) -> str:
    prompt = f"""
    You are a radiologist.

    Visual description: "{caption}"

    **Findings**: Describe the X-ray.

    **Impression**: Provide diagnostic summary.
    """
    result = text_generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]

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

    st.caption("⚠️ Prototype. Not for clinical use.")
