import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient
import torch

hf_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
client = InferenceClient(hf_model)

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip_model()

st.title("🧠 AI Radiology Report Generator")
st.caption("Upload a chest X-ray. The AI will generate a professional radiology report.")

# Upload image
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

def generate_radiology_report(caption: str) -> str:
    prompt = f"""
You are a radiologist assistant AI.

A chest X-ray has been described as follows:
\"{caption}\"

Generate a structured radiology report including:

**Findings**: Describe visual observations from the X-ray.

**Impression**: Summarize diagnostic conclusion.

Use professional clinical terminology.
"""
    response = client.text_generation(prompt=prompt, max_new_tokens=400, temperature=0.7)
    return response.strip()

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
    st.caption("🔬 Note: This is a prototype using BLIP + Hugging Face LLM. Accuracy may vary.")
