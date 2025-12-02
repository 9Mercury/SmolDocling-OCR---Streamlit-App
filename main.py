import streamlit as st
import os
import time
import torch
import tempfile
import re
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Import for Transformers approach
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from huggingface_hub import login
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument
    docling_available = True
except ImportError:
    docling_available = False


def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    if not transformers_available:
        missing.append("transformers huggingface_hub")
    if not docling_available:
        missing.append("docling-core")
    
    return missing


def process_single_image(image, prompt_text="Convert this page to docling."):
    """Process a single image"""
    # Authenticate with Hugging Face
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    start_time = time.time()
    
    # Load processor and model (simplified to avoid potential issues)
    try:
        processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
        model = AutoModelForVision2Seq.from_pretrained(
            "ds4sd/SmolDocling-256M-preview",
            torch_dtype=torch.float32,  # Using simpler dtype
        ).to(device)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise
    
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]
    
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)
    
    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=1024)  # Reduced for testing
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()
    
    # Clean the output
    doctags = doctags.replace("<end_of_utterance>", "").strip()
    
    # Populate document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    
    # Create a docling document
    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)
    
    # Export as markdown
    md_content = doc.export_to_markdown()
    
    processing_time = time.time() - start_time
    
    return doctags, md_content, processing_time


def main():
    st.set_page_config(page_title="SmolDocling OCR App", layout="wide")
    
    st.title("SmolDocling OCR App")
    st.write("Upload images to extract text using SmolDocling OCR")
    
    if not HF_TOKEN:
        st.warning("HF_TOKEN not found in .env file. Authentication may fail.")
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}. Please install them to use this app.")
        st.info("Install with: pip install " + " ".join(missing_deps))
        st.stop()
    
    # Create sidebar
    with st.sidebar:
        st.header("Input Options")
        
        upload_option = st.radio("Choose upload option:", ["Single Image", "Multiple Images"])
        
        task_type = st.selectbox(
            "Select task type",
            [
                "Convert this page to docling.",
                "Convert this table to OTSL.",
                "Convert code to text.",
                "Convert formula to latex.",
                "Convert chart to OTSL.",
                "Extract all section header elements on the page."
            ]
        )
        
        if upload_option == "Single Image":
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", width=250)
        else:
            uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    # Main content area
    if upload_option == "Single Image" and 'uploaded_file' in locals() and uploaded_file is not None:
        process_button = st.button("Process Image")
        
        if process_button:
            with st.spinner("Processing image..."):
                try:
                    doctags, md_content, processing_time = process_single_image(image, task_type)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Extracted DocTags")
                        st.text_area("DocTags Output", doctags, height=400)
                        st.download_button("Download DocTags", doctags, file_name="output.dt")
                    
                    with col2:
                        st.subheader("Markdown Output")
                        st.markdown(md_content)
                        st.download_button("Download Markdown", md_content, file_name="output.md")
                    
                    st.success(f"Processing completed in {processing_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    elif upload_option == "Multiple Images" and 'uploaded_files' in locals() and uploaded_files:
        images = [Image.open(file).convert("RGB") for file in uploaded_files]
        
        if len(images) > 0:
            process_button = st.button("Process Images")
            
            if process_button:
                with st.spinner(f"Processing {len(images)} images..."):
                    try:
                        # Process one by one
                        results = []
                        for idx, image in enumerate(images):
                            st.write(f"Processing image {idx+1}/{len(images)}...")
                            doctags, md_content, processing_time = process_single_image(image, task_type)
                            results.append((doctags, md_content, processing_time))
                        
                        for idx, (doctags, md_content, proc_time) in enumerate(results):
                            with st.expander(f"Image {idx+1} Results"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.image(images[idx], caption=f"Image {idx+1}", width=250)
                                    st.download_button(f"Download DocTags {idx+1}", doctags, file_name=f"output_{idx+1}.dt")
                                
                                with col2:
                                    st.markdown(md_content)
                                    st.download_button(f"Download Markdown {idx+1}", md_content, file_name=f"output_{idx+1}.md")
                            
                            st.write(f"Image {idx+1} processed in {proc_time:.2f} seconds")
                        
                        st.success(f"All images processed successfully")
                    except Exception as e:
                        st.error(f"Error processing images: {str(e)}")
    
    # Information section
    with st.expander("About SmolDocling OCR"):
        st.write("""
        This app uses SmolDocling, a powerful OCR model for document understanding from Hugging Face Hub.
        
        The app extracts DocTags format and converts it to Markdown for easy reading.
        
        Available tasks:
        - Convert pages to DocTags (general OCR)
        - Convert tables to OTSL
        - Convert code snippets to text
        - Convert formulas to LaTeX
        - Convert charts to OTSL
        - Extract section headers
        """)


if __name__ == "__main__":
    main()