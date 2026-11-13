"""
Paathamitra (ಪಾಠಮಿತ್ರ) - Stage 2: Smart Textbook & Photo Reader
==============================================================
In this stage, we add intelligent text extraction for:
1. Native/Digital PDFs (instant direct text extraction using PyMuPDF).
2. Scanned PDFs / Adobe Scan documents (automatic fallback to Groq Vision OCR).
3. Direct smartphone photos (.jpg, .jpeg, .png) via Groq Vision OCR.

Vision OCR Model: meta-llama/llama-4-scout-17b-16e-instruct (Groq Free Tier)
"""

import os
import base64
import streamlit as st
import fitz  # PyMuPDF: fast and lightweight PDF parsing library
from groq import Groq  # Groq SDK for ultra-fast LPU inference
from langchain_core.documents import Document  # Standard container for chunking & vector storage in Stage 3
from dotenv import load_dotenv

# Load environment variables from .env (e.g., GROQ_API_KEY)
load_dotenv()

# Set up page configuration for mobile-friendly view
st.set_page_config(
    page_title="Paathamitra | ಪಾಠಮಿತ್ರ",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Vision OCR Helper (Groq Free Tier)
# ---------------------------------------------------------
def extract_text_from_image_bytes(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """
    Sends an image to Groq Vision (meta-llama/llama-4-scout-17b-16e-instruct)
    to extract English textbook text, mathematical equations, and brief diagram descriptions.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is not set. Please configure your key in the .env file.")

    # Convert binary image bytes to a base64-encoded string for the API payload
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    # Initialize Groq client
    client = Groq(api_key=groq_api_key)
    
    # Instruct Vision LLM to perform accurate textbook OCR and describe diagrams
    prompt = (
        "You are an expert textbook reader and OCR engine. "
        "Transcribe all English text, headings, captions, formulas, tables, and bullet points from this textbook image accurately. "
        "If there are diagrams, illustrations, or charts, provide a concise 1-2 sentence description in brackets like [Diagram: description]. "
        "Output only the transcribed content without any conversational filler or introductory greetings."
    )
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        temperature=0.1,  # Low temperature for strict factual transcription
    )
    
    return response.choices[0].message.content


# ---------------------------------------------------------
# Document Processor (PyMuPDF + Vision OCR Fallback)
# ---------------------------------------------------------
def process_uploaded_file(uploaded_file) -> list[Document]:
    """
    Processes an uploaded file (PDF or Image).
    - Images: processed directly via Groq Vision OCR.
    - PDFs: each page is checked for digital text.
      If a page has < 40 characters (scanned Adobe Scan PDF), it is rendered
      as a PNG at 150 DPI and transcribed using Groq Vision OCR.
    """
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    file_ext = file_name.lower().split(".")[-1]
    documents = []

    # Case A: Smartphone camera photo or direct image upload (.jpg, .jpeg, .png)
    if file_ext in ["jpg", "jpeg", "png"]:
        mime_type = "image/jpeg" if file_ext in ["jpg", "jpeg"] else "image/png"
        with st.spinner("📷 Running Groq Vision OCR on textbook photo..."):
            extracted_text = extract_text_from_image_bytes(file_bytes, mime_type=mime_type)
        
        doc = Document(
            page_content=extracted_text,
            metadata={"source": file_name, "page": 1, "type": "image_ocr"},
        )
        documents.append(doc)

    # Case B: PDF file (Digital textbook or Scanned Adobe Scan PDF)
    elif file_ext == "pdf":
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(pdf_doc)
        
        progress_bar = st.progress(0, text="Processing PDF pages...")

        for index, page in enumerate(pdf_doc):
            page_number = index + 1
            progress_bar.progress(
                (page_number) / total_pages,
                text=f"Reading page {page_number} of {total_pages}...",
            )
            
            # Step 1: Attempt fast digital text extraction
            page_text = page.get_text().strip()

            # Step 2: Smart Detection: If text is less than 40 chars, it's a scanned/photographed page
            if len(page_text) < 40:
                # Render the page to a PNG image at 150 DPI (optimal clarity for OCR without excessive token size)
                pixmap = page.get_pixmap(dpi=150)
                image_bytes = pixmap.tobytes("png")
                
                # Perform Groq Vision OCR on this rendered page
                page_text = extract_text_from_image_bytes(image_bytes, mime_type="image/png")
                page_type = "scanned_ocr"
            else:
                page_type = "digital_text"

            doc = Document(
                page_content=page_text,
                metadata={"source": file_name, "page": page_number, "type": page_type},
            )
            documents.append(doc)
            
        progress_bar.empty()

    return documents


# ---------------------------------------------------------
# Sidebar: Settings & Language Selector
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings (ಸಂಯೋಜನೆಗಳು)")
    
    target_language = st.selectbox(
        label="Explanation Language (ವಿವರಣೆಯ ಭಾಷೆ)",
        options=["Kannada (ಕನ್ನಡ)", "English"],
        index=0,
        help="Choose the language in which you want answers explained.",
    )
    
    st.divider()
    
    # API Key status check helper in sidebar
    groq_key_set = bool(os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") != "your_groq_api_key_here")
    if groq_key_set:
        st.success("🔑 Groq API Key: Detected")
    else:
        st.warning("⚠️ Groq API Key: Not detected in .env")

    st.markdown(
        """
        **About Paathamitra:**
        A ₹0 helper to empower parents to guide children with English textbooks
        using simple, everyday spoken Kannada.
        """
    )

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("📚 Paathamitra (ಪಾಠಮಿತ್ರ)")
st.caption("Empowering parents with textbook explanations in simple spoken Kannada (₹0 API cost)")

# ---------------------------------------------------------
# Section 1: Textbook Chapter / Photo Uploader & Reader
# ---------------------------------------------------------
st.subheader("1. Upload Textbook Chapter or Photo (ಪಠ್ಯಪುಸ್ತಕ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ)")

uploaded_file = st.file_uploader(
    label="Choose a PDF or textbook photo",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Upload digital textbook PDFs, scanned Adobe Scan PDFs, or smartphone camera photos.",
)

if uploaded_file is not None:
    st.success(
        f"✅ File uploaded: **{uploaded_file.name}** "
        f"({round(uploaded_file.size / 1024, 1)} KB)"
    )

    # Reset extracted docs in session_state if a new file is uploaded
    if st.session_state.get("current_file_name") != uploaded_file.name:
        st.session_state["current_file_name"] = uploaded_file.name
        st.session_state["extracted_docs"] = None

    # Extraction Trigger Button
    if st.button("🔍 Extract & Inspect Textbook Content", type="primary"):
        try:
            with st.spinner("Extracting content... Please wait."):
                extracted_docs = process_uploaded_file(uploaded_file)
                st.session_state["extracted_docs"] = extracted_docs
            st.success(f"🎉 Successfully extracted {len(extracted_docs)} page(s)!")
        except Exception as e:
            st.error(f"❌ Error during extraction: {e}")

    # Display preview of extracted pages
    if st.session_state.get("extracted_docs"):
        st.markdown("#### 📖 Extracted Textbook Preview:")
        for doc in st.session_state["extracted_docs"]:
            page_num = doc.metadata.get("page", 1)
            page_type = doc.metadata.get("type", "unknown")
            type_badge = "📷 Scanned / Vision OCR" if "ocr" in page_type else "📄 Digital Text"
            
            with st.expander(f"Page {page_num} ({type_badge})", expanded=True):
                st.text_area(
                    label=f"Text content for Page {page_num}",
                    value=doc.page_content,
                    height=180,
                    disabled=True,
                    key=f"preview_page_{page_num}",
                )

st.divider()

# ---------------------------------------------------------
# Section 2: Question & Verification Echo Box
# ---------------------------------------------------------
st.subheader("2. Ask a Question (ಪ್ರಶ್ನೆ ಕೇಳಿ)")

user_question = st.text_input(
    label="Enter your question about the uploaded lesson:",
    placeholder="e.g., ದ್ಯುತಿಸಂಶ್ಲೇಷಣೆ ಅಂದ್ರೆ ಏನು? or What is photosynthesis?",
)

if user_question:
    st.markdown("### 💬 Your Question (ನಿಮ್ಮ ಪ್ರಶ್ನೆ):")
    st.write(f"**Selected Language:** {target_language}")
    st.write(f"**Echo:** {user_question}")
    st.info("ℹ️ Vector database chunking will be added in Stage 3, and AI answers in Stage 4.")
