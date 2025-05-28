import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load BLIP 
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

st.title(" AI Radiology Report Generator")
st.caption("Upload a chest X-ray. The AI will generate a clinical-style radiology report.")

# Upload image
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

# Function to generate the report using GPT
def generate_radiology_report(caption: str) -> str:
    prompt = f"""
You are a radiologist assistant AI.

A chest X-ray has been described as follows:
"{caption}"

Generate a structured radiology report including:

**Findings**: Describe the visual details of the image.

**Impression**: Provide a diagnostic conclusion or summary.

Use professional medical terminology appropriate for a clinical radiology report.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Or "gpt-3.5-turbo" if you don't have GPT-4 access
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=400
    )
    return response.choices[0].message["content"]

# Processing uploaded image
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Chest X-ray", use_column_width=True)

    # Captioning with BLIP
    with st.spinner("🔍 Analyzing Image..."):
        inputs = processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.markdown(f"🧠 **Visual Description:** _{caption}_")

    # Report generation
    with st.spinner("📝 Generating Report..."):
        report = generate_radiology_report(caption)
        st.markdown("### 📄 Generated Radiology Report")
        st.write(report)

    st.markdown("---")
    st.caption("🔬 Note: This is a prototype combining vision + LLM. Accuracy may vary.")
