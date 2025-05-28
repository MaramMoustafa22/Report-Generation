import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient

# Load BLIP model and processor for caption generation
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize InferenceClient for text generation
client = InferenceClient(
    model="tiiuae/falcon-7b-instruct",
    token=st.secrets["HF_TOKEN"]
)

def generate_caption(image: Image.Image) -> str:
    inputs = processor(images=image, return_tensors="pt")
    outputs = blip_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def generate_radiology_report(caption: str) -> str:
    prompt = f"""
You are an expert radiologist. Based on the following visual description, write a structured radiology report.

**Visual Description**: {caption}

**Findings**: Describe anatomical observations.

**Impression**: Provide diagnostic summary.
"""
    try:
        response = client.text_generation(prompt=prompt, max_new_tokens=400, temperature=0.7)
        return response.strip()
    except Exception as e:
        st.error(f"❌ Failed to generate report: {e}")
        return "Report generation failed."

# Streamlit UI
st.set_page_config(page_title="Radiology Report Generator", layout="centered")
st.title("🩻 AI Radiology Report Generator")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray or Ultrasound Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🧠 Generating Caption..."):
        caption = generate_caption(image)
    st.markdown(f"🧠 **Visual Description:** _{caption}_")

    with st.spinner("📝 Generating Report using Falcon-7B..."):
        report = generate_radiology_report(caption)
    st.markdown("### 📄 Generated Radiology Report")
    st.write(report)
