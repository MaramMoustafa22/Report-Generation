import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient

# Load BLIP model (image captioning)
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Hugging Face Inference Client (text generation)
@st.cache_resource
def load_hf_client():
    return InferenceClient(token=st.secrets["hf_token"])

# Generate caption
def generate_caption(image):
    processor, model = load_caption_model()
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# Generate report
def generate_report(caption):
    prompt = f"""You are a radiology assistant. Generate a short radiology report based on the image description.

**Findings**: {caption}

**Impression**: Diagnostic summary.
"""
    client = load_hf_client()
    return client.text_generation(prompt, model="tiiuae/falcon-7b-instruct", max_new_tokens=300)

# UI
st.set_page_config(page_title="Radiology Report Generator", layout="centered")
st.title("🩻 Radiology Report Generator")

uploaded_file = st.file_uploader("Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating visual description..."):
        caption = generate_caption(image)
        st.success("Caption Generated!")
        st.markdown(f"**🧠 Description:** _{caption}_")

    with st.spinner("Generating radiology report..."):
        try:
            report = generate_report(caption)
            st.markdown("### 📄 Radiology Report")
            st.write(report.strip())
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
