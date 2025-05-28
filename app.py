import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP model for image captioning 
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

st.title("Radiology Report Generator")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)


    prompt = f"""
    You are a radiologist AI assistant. Based on the following visual findings from a chest X-ray, generate a professional radiology report including both 'Findings' and 'Impression'.

    Visual description: {caption}
    
    Please write a clinically styled report.
    """


    dummy_report = f"""
    **Findings**: The lungs are clear. The heart size is within normal limits. No evidence of pleural effusion or pneumothorax. Bony structures are intact.

    **Impression**: Normal chest X-ray.
    """

    st.markdown("### 📝 Generated Radiology Report:")
    st.write(dummy_report)

    st.markdown("---")
    st.caption("Note: This is a prototype using BLIP captioning. Final model will generate from medical training.")
