# Project Specification: ₹0 Multilingual Parent Study Helper (Kannada & English RAG)

## 1. Project Overview
Build a mobile-friendly web application using **Python** and **Streamlit** that helps non-English speaking parents assist their children with English school textbooks at **₹0 API cost**.
- **Input:** Parents upload a child's textbook chapter (digital PDF, Adobe Scan PDF, or a direct smartphone photo `.jpg`/`.png`).
- **Querying:** Parents ask questions via **Kannada/Kanglish voice** (microphone) or text input.
- **Output:** The app retrieves the exact relevant English textbook excerpts and explains the concept in **simple, everyday spoken Kannada** (keeping key English scientific/academic terms in brackets) with optional Kannada audio playback.

---

## 2. CRITICAL AGENT INSTRUCTIONS (Incremental Learning Mode)
**IMPORTANT — Follow these rules strictly:**
1. **Do NOT generate the entire `app.py` file all at once.**
2. Build the project **one stage at a time** (Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5).
3. At each stage:
   - Write only the code needed for that specific stage.
   - Add clear, beginner-friendly inline comments explaining *why* each library and function is used.
   - Give a brief plain-English summary of what changed and how the user can test that stage locally.
   - **STOP and ask the user for confirmation** before moving to the next stage.

---

## 3. Tech Stack & ₹0 Models
- **Frontend UI:** Streamlit (`st.audio_input`, `st.file_uploader`, mobile-responsive layout)
- **Orchestration:** LangChain (`langchain-core`, `langchain-community`, `langchain-text-splitters`)
- **PDF & Image Processing:** `PyMuPDF` (`fitz`) to extract text from digital PDFs and render scanned PDF pages into PNG images for Vision OCR.
- **Vision OCR (Scanned PDFs & Photos):** Groq Vision API (`meta-llama/llama-4-scout-17b-16e-instruct` — Free Tier)
- **Local Multilingual Embeddings:** `HuggingFaceEmbeddings` using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (runs locally on CPU for ₹0)
- **Local Vector Database:** `Chroma` (`langchain-chroma`), persisted locally to `./textbook_db`
- **Reasoning LLM:** `ChatGroq` (`llama-3.3-70b-versatile` — Free Tier)
- **Kannada Speech-to-Text (STT):** Groq Whisper API (`whisper-large-v3-turbo` with `language="kn"` — Free Tier)
- **Kannada Text-to-Speech (TTS):** `edge-tts` Python library using Microsoft's free neural Kannada voice (`kn-IN-SapnaNeural` — ₹0, no API key required)

---

## 4. Directory Structure
```text
kannada-study-helper/
├── app.py                  # Main Streamlit application (built incrementally across 5 stages)
├── requirements.txt        # Python dependencies
├── .env.example            # Template for GROQ_API_KEY
├── .gitignore              # Ignore .env, venv/, __pycache__/, and textbook_db/
└── README.md               # Setup, local run, and Hugging Face Spaces deployment guide
```

---

## 5. Step-by-Step Implementation Stages

### Stage 1: Project Setup & Basic Streamlit UI Skeleton
- **Goal:** Set up the folder, dependencies, and a working barebones Streamlit screen.
- **Tasks:**
  1. Create `requirements.txt` with:
     `streamlit`, `python-dotenv`, `pymupdf`, `groq`, `edge-tts`, `sentence-transformers`, `langchain`, `langchain-community`, `langchain-huggingface`, `langchain-chroma`, `langchain-groq`
  2. Create `.env.example` containing `GROQ_API_KEY="your_groq_api_key_here"` and `.gitignore` ignoring `.env`, `venv/`, `__pycache__/`, and `textbook_db/`.
  3. Create a starter `app.py` with the page title, sidebar language selector (`"Kannada (ಕನ್ನಡ)"`, `"English"`), file uploader (`["pdf", "png", "jpg", "jpeg"]`), and text input box that echoes back user input so the user can verify Streamlit runs via `streamlit run app.py`.
- **Pause & Confirm:** Explain the project files created and wait for user confirmation.

### Stage 2: Smart Textbook & Photo Reader (PyMuPDF + Groq Vision OCR)
- **Goal:** Extract clean English text from digital PDFs, Adobe Scan PDFs, and phone photos.
- **Tasks:**
  1. Add `extract_text_from_image_bytes(image_bytes, mime_type)` using Groq Vision (`meta-llama/llama-4-scout-17b-16e-instruct`) to extract text, formulas, and brief 1-2 sentence diagram descriptions.
  2. Add `process_uploaded_file(uploaded_file)` using `fitz` (`PyMuPDF`):
     - If `.jpg`/`.jpeg`/`.png`, run Vision OCR directly.
     - If `.pdf`, check each page's digital text length; if `< 40` characters (scanned page), render at `150 DPI` to PNG and run Vision OCR.
  3. Display a preview of the extracted text in the UI so the user can test uploading a photo or scanned PDF.
- **Pause & Confirm:** Explain how digital vs. scanned PDF detection works and wait for user confirmation.

### Stage 3: Chunking, Multilingual Embeddings & Vector Database (ChromaDB)
- **Goal:** Store extracted textbook pages in a searchable local vector database.
- **Tasks:**
  1. Add `@st.cache_resource` function `get_embedding_model()` loading `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` locally on CPU.
  2. Update `process_uploaded_file()` to split extracted `Document` pages using `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` and save vectors into `./textbook_db` using `Chroma`.
  3. Add a **"Clear Textbook Memory"** button in the sidebar to delete `./textbook_db` when switching chapters/subjects.
- **Pause & Confirm:** Explain how multilingual embeddings match Kannada queries to English text and wait for user confirmation.

### Stage 4: LangChain RAG Pipeline (Grounded Kannada Explanations)
- **Goal:** Answer typed Kannada or English questions using retrieved textbook chunks.
- **Tasks:**
  1. Add `ask_textbook(question, target_language)` retrieving top `k=3` chunks from `./textbook_db`.
  2. Build an LCEL chain (`prompt | llm | StrOutputParser()`) using `ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)`.
  3. Instruct the prompt to answer **only** from textbook context, explain in simple everyday spoken Kannada, keep key English textbook words in brackets (e.g., `ದ್ಯುತಿಸಂಶ್ಲೇಷಣೆ (Photosynthesis)`), and cite page numbers.
  4. Display the answer plus an `st.expander("📖 View Extracted English Textbook Text")` showing source chunks.
- **Pause & Confirm:** Explain the LCEL chain (`|`) and prompt grounding rules, then wait for user confirmation.

### Stage 5: Free Kannada Voice Input & Output (Groq Whisper + Edge-TTS)
- **Goal:** Enable hands-free Kannada voice questions and spoken audio replies at ₹0 cost.
- **Tasks:**
  1. Add `st.audio_input("🎤 Tap to speak in Kannada or English")` to the main UI.
  2. Add `transcribe_kannada_audio(audio_bytes)` using `groq_client.audio.transcriptions.create(model="whisper-large-v3-turbo", language="kn")` to convert spoken Kannada/Kanglish into text.
  3. Add `speak_kannada_text(text)` using `edge_tts.Communicate(text[:500], voice="kn-IN-SapnaNeural")` (with a sidebar checkbox `"Read answer aloud"`, default `False`) and play the generated MP3 via `st.audio()`.
  4. Create `README.md` with local run instructions and Hugging Face Spaces deployment steps.
- **Pause & Confirm:** Summarize the complete end-to-end ₹0 application.
