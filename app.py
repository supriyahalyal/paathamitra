"""
Paathamitra (ಪಾಠಮಿತ್ರ) - Stage 5: Free Kannada Voice Input & Output (Groq Whisper + Edge-TTS)
=============================================================================================
In this final stage, we complete the end-to-end ₹0 Multilingual Parent Study Helper:
1. Speech-to-Text (STT): Groq Whisper API (whisper-large-v3-turbo with language="kn")
   converts spoken Kannada/Kanglish into accurate text via st.audio_input.
2. Text-to-Speech (TTS): Microsoft neural voice (kn-IN-SapnaNeural) via edge-tts
   (100% free, no API key required) to read explanations aloud for parents.
3. Audio Controls: Sidebar toggle ("Read answer aloud") and on-demand audio playback button.
4. Full RAG integration: Upload -> OCR -> Vector DB -> Retrieval -> Reasoning -> Spoken Audio.
"""

import os
import sys
import shutil
import base64
import asyncio
import streamlit as st
import fitz  # PyMuPDF: fast and lightweight PDF parsing library
from groq import Groq  # Groq SDK for ultra-fast LPU inference (Vision, LLM, Whisper)
from langchain_core.documents import Document  # Container for page & chunk text with metadata
from langchain_core.embeddings import Embeddings  # Base interface for vector embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Smart text chunking
from langchain_core.prompts import ChatPromptTemplate  # Prompt templating
from langchain_core.output_parsers import StrOutputParser  # Clean string parser for LCEL
from langchain_groq import ChatGroq  # Groq LLM integration
import edge_tts  # Microsoft Neural TTS (100% free, ₹0 API key)
from dotenv import load_dotenv

# Try importing Chroma with safe fallbacks
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Load environment variables from .env
load_dotenv()

# Set up page configuration for mobile-friendly view
st.set_page_config(
    page_title="Paathamitra | ಪಾಠಮಿತ್ರ",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

DB_DIRECTORY = "./textbook_db"


# ---------------------------------------------------------
# FastEmbed Wrapper (ONNX Runtime - Microsoft Signed DLLs)
# ---------------------------------------------------------
class FastEmbedWrapper(Embeddings):
    """
    Wraps Qdrant's FastEmbed into LangChain's Embeddings interface.
    Uses Microsoft-signed ONNX runtime, which is immune to Windows
    Application Control / Smart App Control DLL blocking policies.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vec)) for vec in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, next(self.model.embed([text]))))


# ---------------------------------------------------------
# Local Multilingual Embedding Model (CPU, ₹0 Cost)
# ---------------------------------------------------------
@st.cache_resource
def get_embedding_model():
    """
    Loads sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 locally on CPU.
    1. Tries FastEmbed (lightweight ONNX runtime with Microsoft-signed binaries).
    2. Falls back to sentence-transformers / PyTorch.
    """
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Priority 1: FastEmbed (Robust on Windows 11 with Smart App Control)
    try:
        import fastembed
        return FastEmbedWrapper(model_name=model_name)
    except Exception:
        pass

    # Priority 2: Standard sentence-transformers (PyTorch)
    try:
        import sentence_transformers
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as hf_err:
        st.error(
            f"### 🛡️ Windows Security Blocked PyTorch Binary\n\n"
            f"**Error Details:** `{type(hf_err).__name__}: {hf_err}`\n\n"
            f"Windows Application Control or Smart App Control blocked PyTorch's unsigned `_C.pyd` DLL.\n\n"
            f"**Instant Solution:** Install `fastembed` (uses Microsoft-signed ONNX binaries):\n"
            f"```powershell\n"
            f"& \"{sys.executable}\" -m pip install fastembed\n"
            f"```"
        )
        raise hf_err


# ---------------------------------------------------------
# Vector Database Management Helpers
# ---------------------------------------------------------
def is_database_populated() -> bool:
    """Checks if Chroma vector database directory exists and contains data files."""
    return os.path.exists(DB_DIRECTORY) and bool(os.listdir(DB_DIRECTORY))


def get_vector_store():
    """Loads existing Chroma vector store if it exists on disk."""
    if is_database_populated():
        try:
            return Chroma(
                persist_directory=DB_DIRECTORY,
                embedding_function=get_embedding_model(),
            )
        except Exception:
            return None
    return None


def clear_vector_store():
    """Deletes local Chroma database directory and clears session memory."""
    try:
        if os.path.exists(DB_DIRECTORY):
            vs = get_vector_store()
            if vs is not None:
                try:
                    vs.delete_collection()
                except Exception:
                    pass
            shutil.rmtree(DB_DIRECTORY, ignore_errors=True)
    except Exception as e:
        st.warning(f"Note while clearing database: {e}")

    st.session_state["extracted_docs"] = None
    st.session_state["chunks_count"] = 0
    st.session_state["current_file_name"] = None
    st.session_state["last_transcription"] = None


# ---------------------------------------------------------
# Vision OCR Helper (Groq Free Tier with Dynamic Fallback)
# ---------------------------------------------------------
def extract_text_from_image_bytes(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """
    Sends an image to Groq Vision to extract English textbook text,
    mathematical equations, and brief diagram descriptions.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is not set. Please configure your key in the .env file.")

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    client = Groq(api_key=groq_api_key)
    
    candidate_models = []
    try:
        available_ids = [m.id for m in client.models.list().data]
        priority_models = [
            "qwen/qwen3.8-27b",
            "meta-models/muse-spark-open",
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ]
        for p in priority_models:
            if p in available_ids and p not in candidate_models:
                candidate_models.append(p)
                
        for m_id in available_ids:
            if any(k in m_id.lower() for k in ["vision", "vl", "scout"]) and m_id not in candidate_models:
                candidate_models.append(m_id)
    except Exception:
        pass

    for fallback in ["qwen/qwen3.8-27b", "meta-models/muse-spark-open", "llama-3.2-11b-vision-preview"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    prompt = (
        "You are an expert textbook reader and OCR engine. "
        "Transcribe all English text, headings, captions, formulas, tables, and bullet points from this textbook image accurately. "
        "If there are diagrams, illustrations, or charts, provide a concise 1-2 sentence description in brackets like [Diagram: description]. "
        "Output only the transcribed content without any conversational filler or introductory greetings."
    )

    last_error = None
    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
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
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            err_msg = str(e).lower()
            if "model_not_found" in err_msg or "does not exist" in err_msg or "404" in err_msg:
                continue
            else:
                raise e

    raise RuntimeError(
        f"Unable to process image. Tried vision models {candidate_models}. Error: {last_error}"
    )


# ---------------------------------------------------------
# Document Processor & Vector Store Indexer
# ---------------------------------------------------------
def process_uploaded_file(uploaded_file) -> tuple[list[Document], int]:
    """
    1. Extracts text from PDF or Image using PyMuPDF / Groq Vision OCR.
    2. Chunks extracted Document pages using RecursiveCharacterTextSplitter.
    3. Generates local multilingual embeddings and indexes into ./textbook_db using Chroma.
    """
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    file_ext = file_name.lower().split(".")[-1]
    raw_documents = []

    # Case A: Smartphone camera photo or direct image upload (.jpg, .jpeg, .png)
    if file_ext in ["jpg", "jpeg", "png"]:
        mime_type = "image/jpeg" if file_ext in ["jpg", "jpeg"] else "image/png"
        with st.spinner("📷 Running Groq Vision OCR on textbook photo..."):
            extracted_text = extract_text_from_image_bytes(file_bytes, mime_type=mime_type)
        
        doc = Document(
            page_content=extracted_text,
            metadata={"source": file_name, "page": 1, "type": "image_ocr"},
        )
        raw_documents.append(doc)

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
            
            page_text = page.get_text().strip()

            if len(page_text) < 40:
                pixmap = page.get_pixmap(dpi=150)
                image_bytes = pixmap.tobytes("png")
                page_text = extract_text_from_image_bytes(image_bytes, mime_type="image/png")
                page_type = "scanned_ocr"
            else:
                page_type = "digital_text"

            doc = Document(
                page_content=page_text,
                metadata={"source": file_name, "page": page_number, "type": page_type},
            )
            raw_documents.append(doc)
            
        progress_bar.empty()

    # Step 2: Chunking with RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(raw_documents)

    # Step 3: Embed & Store in local ChromaDB
    with st.spinner("🧠 Generating multilingual embeddings and indexing to ./textbook_db..."):
        embedding_model = get_embedding_model()
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=DB_DIRECTORY,
        )

    return raw_documents, len(chunks)


# ---------------------------------------------------------
# Dynamic Groq Reasoning Model Discovery
# ---------------------------------------------------------
def get_groq_reasoning_models(groq_api_key: str) -> list[str]:
    """
    Finds available text reasoning models on Groq in order of preference.
    Ensures that 404 model_not_found errors are avoided.
    """
    priority = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "qwen/qwen3-32b",
        "qwen/qwen3.8-27b",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
    ]
    candidates = []
    try:
        client = Groq(api_key=groq_api_key)
        available_ids = [m.id for m in client.models.list().data]
        for p in priority:
            if p in available_ids and p not in candidates:
                candidates.append(p)
        for m_id in available_ids:
            if not any(k in m_id.lower() for k in ["whisper", "guard", "vision", "preview"]) and m_id not in candidates:
                candidates.append(m_id)
    except Exception:
        pass

    for p in priority:
        if p not in candidates:
            candidates.append(p)
    return candidates


# ---------------------------------------------------------
# Stage 4: LangChain RAG Question Answering Function
# ---------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are 'Paathamitra' (ಪಾಠಮಿತ್ರ), an empathetic, encouraging educational tutor helping parents assist their children with school lessons.

Your mission:
Explain the textbook concepts so clearly and simply that even a parent with no prior English or science background can understand it and explain it to their child.

Strict Grounding Rules:
1. Answer ONLY using the provided Textbook Excerpts. Do NOT use outside knowledge or hallucinate.
2. If the question cannot be answered from the provided excerpts, politely explain:
   - In Kannada: "ಕ್ಷಮಿಸಿ, ನೀವು ಕೇಳಿದ ವಿಷಯ ಈ ಪಠ್ಯಪುಸ್ತಕದ ಭಾಗದಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ."
   - In English: "Sorry, this topic is not covered in the uploaded textbook chapter."
3. Language and Terminology Rules:
   - If Target Language is "Kannada (ಕನ್ನಡ)":
     * Use simple, warm, everyday spoken Kannada (ಹಳ್ಳಿ/ಮನೆಯಲ್ಲಿ ಮಾತನಾಡುವ ಸರಳ ಕನ್ನಡ).
     * CRITICAL: Whenever introducing a scientific or academic concept in Kannada, ALWAYS write the original English textbook term in brackets immediately next to it.
       Examples: ದ್ಯುತಿಸಂಶ್ಲೇಷಣೆ (Photosynthesis), ಹೂವಿನ ಎಸಳುಗಳು (Petals), ಕೇಸರ (Stamen), ಅಂಡಾಶಯ (Ovary), ಪರಾಗಸ್ಪರ್ಶ (Pollination).
   - If Target Language is "English":
     * Explain in plain, friendly, conversational English suitable for a parent.
4. Citing Sources:
   * Conclude your answer by citing the exact page numbers from which the information was gathered (e.g., "📖 ಆಧಾರ: ಪುಟ 2, 3" or "📖 Source: Page 2, 3").
"""

RAG_USER_PROMPT = """Textbook Excerpts:
{context}

Target Explanation Language: {target_language}
Parent's Question: {question}

Please explain the answer for the parent:"""


def ask_textbook(question: str, target_language: str) -> tuple[str, list[Document]]:
    """
    Retrieves top k=3 chunks from ChromaDB and executes an LCEL chain:
    prompt | llm | StrOutputParser() with automatic model fallback.
    """
    vs = get_vector_store()
    if vs is None:
        raise ValueError("Vector database not found. Please upload and index a textbook chapter first.")

    retrieved_docs = vs.similarity_search(question, k=3)
    if not retrieved_docs:
        return "No relevant textbook content found in the database.", []

    context_parts = []
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", "Unknown")
        context_parts.append(f"[Textbook Page {page_num}]:\n{doc.page_content}")
    context_text = "\n\n---\n\n".join(context_parts)

    groq_api_key = os.getenv("GROQ_API_KEY")
    candidate_models = get_groq_reasoning_models(groq_api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_USER_PROMPT),
    ])

    last_error = None
    for model_name in candidate_models:
        try:
            llm = ChatGroq(
                model=model_name,
                temperature=0.3,
                groq_api_key=groq_api_key,
            )
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "context": context_text,
                "target_language": target_language,
                "question": question,
            })
            return answer, retrieved_docs
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                continue
            else:
                raise e

    raise RuntimeError(
        f"Unable to generate explanation. Tried models {candidate_models}. Last error: {last_error}"
    )


# ---------------------------------------------------------
# Stage 5: Kannada Speech-to-Text (Groq Whisper API)
# ---------------------------------------------------------
def transcribe_kannada_audio(audio_bytes: bytes, target_language: str = "Kannada (ಕನ್ನಡ)") -> str:
    """
    Transcribes spoken Kannada, Kanglish, or English audio into text
    using Groq Whisper API (whisper-large-v3-turbo, language="kn" or "en").
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is not set. Please configure your key in the .env file.")

    client = Groq(api_key=groq_api_key)
    lang_code = "kn" if "Kannada" in target_language else "en"

    # Send audio bytes directly in tuple format (filename, bytes) to Groq Whisper
    transcription = client.audio.transcriptions.create(
        file=("user_audio.wav", audio_bytes),
        model="whisper-large-v3-turbo",
        language=lang_code,
        response_format="text",
    )
    
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()


# ---------------------------------------------------------
# Stage 5: Kannada Text-to-Speech (Edge-TTS kn-IN-SapnaNeural)
# ---------------------------------------------------------
async def _generate_edge_tts_audio(text: str, voice: str) -> bytes:
    """Asynchronous worker to stream audio bytes from Microsoft Edge TTS."""
    communicate = edge_tts.Communicate(text, voice)
    audio_stream = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream += chunk["data"]
    return audio_stream


def speak_kannada_text(text: str, target_language: str = "Kannada (ಕನ್ನಡ)") -> bytes:
    """
    Synthesizes natural Kannada speech using Microsoft Neural Voice (kn-IN-SapnaNeural)
    at ₹0 API cost with zero external registration required.
    Truncates to the first 500 characters for snappy, instantaneous audio delivery.
    """
    # Select voice based on language setting
    voice = "kn-IN-SapnaNeural" if "Kannada" in target_language else "en-IN-NeerjaNeural"
    
    # Take first 500 characters to keep audio response snappy
    cleaned = text[:500]
    # Remove markdown formatting characters for clean spoken pronunciation
    for ch in ["*", "#", "_", "`", ">", "[", "]", "(", ")"]:
        cleaned = cleaned.replace(ch, " ")
    cleaned = " ".join(cleaned.split())

    try:
        # Run async coroutine safely across environments
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If running inside an existing loop, run via new event loop
                new_loop = asyncio.new_event_loop()
                return new_loop.run_until_complete(_generate_edge_tts_audio(cleaned, voice))
            return loop.run_until_complete(_generate_edge_tts_audio(cleaned, voice))
        except RuntimeError:
            return asyncio.run(_generate_edge_tts_audio(cleaned, voice))
    except Exception as e:
        st.warning(f"Could not synthesize audio: {e}")
        return b""


# ---------------------------------------------------------
# Sidebar: Settings, Model Status & Clear Memory Button
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings (ಸಂಯೋಜನೆಗಳು)")
    
    target_language = st.selectbox(
        label="Explanation Language (ವಿವರಣೆಯ ಭಾಷೆ)",
        options=["Kannada (ಕನ್ನಡ)", "English"],
        index=0,
        help="Choose the language in which you want answers explained.",
    )
    
    # Audio Read Aloud Toggle
    read_aloud = st.checkbox(
        label="🔊 Read answer aloud (ಧ್ವನಿ ಮೂಲಕ ಓದಿ)",
        value=False,
        help="Automatically plays spoken Kannada audio narration using Microsoft's free neural voice (kn-IN-SapnaNeural).",
    )

    st.divider()

    # Memory Status Indicator
    st.subheader("📚 Memory Management")
    if is_database_populated():
        st.success("🟢 Active Textbook Memory: Ready")
        if st.button("🗑️ Clear Textbook Memory", help="Deletes stored chapter embeddings to load a new lesson/subject"):
            clear_vector_store()
            st.success("🧹 Textbook memory cleared! You can now upload a new lesson.")
            st.rerun()
    else:
        st.info("⚪ Textbook Memory: Empty")

    st.divider()
    
    # API Key & Model status check in sidebar
    groq_key_set = bool(os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") != "your_groq_api_key_here")
    if groq_key_set:
        st.success("🔑 Groq API Key: Detected")
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            available_models = [m.id for m in client.models.list().data]
            
            vision_candidates = [m for m in available_models if any(k in m.lower() for k in ["qwen", "muse", "vision", "vl", "scout"])]
            if vision_candidates:
                st.caption(f"👁️ Vision Model: `{vision_candidates[0]}`")
                
            reasoning_candidates = get_groq_reasoning_models(os.getenv("GROQ_API_KEY"))
            if reasoning_candidates:
                st.caption(f"🤖 Reasoning Model: `{reasoning_candidates[0]}`")
        except Exception:
            pass
    else:
        st.warning("⚠️ Groq API Key: Not detected in .env")

    st.caption("🧠 Embedding: `paraphrase-multilingual-MiniLM-L12-v2`")
    st.caption("🎤 STT: `whisper-large-v3-turbo` (Groq Free)")
    st.caption("🔊 TTS: `kn-IN-SapnaNeural` (Edge-TTS ₹0)")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("📚 Paathamitra (ಪಾಠಮಿತ್ರ)")
st.caption("Empowering parents with textbook explanations in simple spoken Kannada (₹0 API cost)")

# Notification if textbook is already loaded from disk
if is_database_populated() and not st.session_state.get("extracted_docs"):
    st.info("💡 **Textbook chapter is already loaded in local memory!** You do not need to re-upload. You can ask questions directly in Section 2 below.")

# ---------------------------------------------------------
# Section 1: Textbook Chapter / Photo Uploader & Indexer
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

    if st.session_state.get("current_file_name") != uploaded_file.name:
        st.session_state["current_file_name"] = uploaded_file.name
        st.session_state["extracted_docs"] = None
        st.session_state["chunks_count"] = 0

    if st.button("🔍 Extract & Build Vector Database", type="primary"):
        try:
            extracted_docs, chunk_count = process_uploaded_file(uploaded_file)
            st.session_state["extracted_docs"] = extracted_docs
            st.session_state["chunks_count"] = chunk_count
            st.success(f"🎉 Successfully extracted {len(extracted_docs)} page(s) and stored {chunk_count} chunks into `./textbook_db`!")
        except Exception as e:
            st.error(f"❌ Error during processing: {e}")

    if st.session_state.get("extracted_docs"):
        st.markdown(f"#### 📖 Extracted Pages & Chunks ({st.session_state.get('chunks_count', 0)} chunks indexed):")
        for doc in st.session_state["extracted_docs"]:
            page_num = doc.metadata.get("page", 1)
            page_type = doc.metadata.get("type", "unknown")
            type_badge = "📷 Scanned / Vision OCR" if "ocr" in page_type else "📄 Digital Text"
            
            with st.expander(f"Page {page_num} ({type_badge})", expanded=False):
                st.text_area(
                    label=f"Text content for Page {page_num}",
                    value=doc.page_content,
                    height=180,
                    disabled=True,
                    key=f"preview_page_{page_num}",
                )

st.divider()

# ---------------------------------------------------------
# Section 2: Grounded RAG Question Answering (Voice & Text)
# ---------------------------------------------------------
st.subheader("2. Ask a Question (ಪ್ರಶ್ನೆ ಕೇಳಿ)")

# Voice Microphone Input via Streamlit 1.40+
audio_input_data = st.audio_input("🎤 Tap to speak in Kannada or English (ಧ್ವನಿ ಮೂಲಕ ಪ್ರಶ್ನೆ ಕೇಳಿ)")

voice_transcription = ""
if audio_input_data is not None:
    audio_bytes = audio_input_data.getvalue()
    # Check if this is a newly recorded audio clip
    if st.session_state.get("last_recorded_audio") != audio_bytes:
        st.session_state["last_recorded_audio"] = audio_bytes
        with st.spinner("🎙️ Transcribing Kannada speech using Groq Whisper..."):
            try:
                voice_transcription = transcribe_kannada_audio(audio_bytes, target_language)
                st.session_state["active_transcription"] = voice_transcription
                st.success(f"🗣️ Heard: **{voice_transcription}**")
            except Exception as e:
                st.error(f"❌ Voice transcription error: {e}")

# Text input fallback / manual typing
user_typed_question = st.text_input(
    label="Or type your question in Kannada, Kanglish, or English:",
    value=st.session_state.get("active_transcription", ""),
    placeholder="e.g., ಹೂವಿನ ಭಾಗಗಳು ಯಾವುವು? or What are the parts of a flower?",
)

active_question = user_typed_question.strip()

if active_question:
    st.markdown("### 💬 Your Question (ನಿಮ್ಮ ಪ್ರಶ್ನೆ):")
    st.write(f"**Question:** {active_question}")
    st.write(f"**Selected Explanation Language:** {target_language}")
    
    vs = get_vector_store()
    if vs is not None:
        with st.spinner("🤖 Consulting Paathamitra & reading textbook excerpts..."):
            try:
                answer, source_docs = ask_textbook(active_question, target_language)
                
                # Display Answer in a clean presentation card
                st.markdown("### 💡 Explanation (ವಿವರಣೆ):")
                st.markdown(answer)
                
                # Audio Playback Handling
                # 1. Automatic if sidebar checkbox is ticked
                # 2. Or manual button click
                col1, col2 = st.columns([1, 4])
                with col1:
                    play_clicked = st.button("🔊 Play Voice (ಕೇಳಿ)")

                if read_aloud or play_clicked:
                    with st.spinner("🔊 Generating spoken Kannada audio..."):
                        audio_speech = speak_kannada_text(answer, target_language)
                        if audio_speech:
                            st.audio(audio_speech, format="audio/mp3", autoplay=True)
                
                # Display Source Chunks Expander for full parent transparency
                with st.expander("📖 View Extracted English Textbook Text (ಮೂಲ ಪಠ್ಯಪುಸ್ತಕ ಪುಟಗಳು)", expanded=False):
                    for i, doc in enumerate(source_docs, 1):
                        page_ref = doc.metadata.get("page", "Unknown")
                        st.markdown(f"**Excerpt {i} (from Page {page_ref}):**")
                        st.info(doc.page_content)
                        
            except Exception as e:
                st.error(f"❌ Error generating explanation: {e}")
    else:
        st.warning("⚠️ No textbook indexed yet. Please upload a chapter and click 'Extract & Build Vector Database' above.")
