import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
import torch

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

@st.cache_resource
def load_text_generator():
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")

text_generator = load_text_generator()

st.title("🧠 AI Radiology Report Generator")
st.caption("Upload a chest X-ray. The AI will generate a radiology report using image caption + text generation.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

def generate_radiology_report(caption: str) -> str:
    prompt = f"""You are a professional radiologist.

A chest X-ray shows: "{caption}"

Write a radiology report:

**Findings**: 
**Impression**: 
"""
    result = text_generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    return result[0]["generated_text"].replace(prompt, "").strip()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Chest X-ray", use_column_width=True)

    with st.spinner("🔍 Analyzing Image..."):
        inputs = processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.markdown(f"🧠 **Visual Description:** _{caption}_")

    with st.spinner("📝 Generating Report..."):
        report = generate_radiology_report(caption)
        st.markdown("### 📄 Generated Radiology Report")
        st.write(report)

    st.markdown("---")
    st.caption("⚠️ Note: This prototype uses BLIP and Falcon-7B. Output is not a medical diagnosis.")
