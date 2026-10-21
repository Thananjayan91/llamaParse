# 🚀 LlamaParse & RAG Studio (Flask + LlamaCloud + LlamaIndex)

An end-to-end web application built with **Flask**, **Jinja2 templates (`render_template`)**, **LlamaCloud (v2 LlamaParse API & Structured Extraction)**, and **LlamaIndex RAG (Retrieval-Augmented Generation)**.

---

## 🌟 Key Features

1. **📑 LlamaParse Document Ingestion**:
   - Upload any document (PDF, DOCX, PPTX, TXT, PNG, JPG, CSV, XLSX).
   - Select parsing tiers: `agentic_plus` (Recommended), `agentic`, `cost_effective`, `fast`.
   - Real-time conversion to layout-aware **Full Markdown (`.md`)** and **Clean Text (`.txt`)**.
   - Interactive live markdown rendered preview with code highlight and download buttons.

2. **📊 Structured Data Extraction**:
   - Schema-based JSON extraction from unstructured documents.
   - Built-in schema templates (Invoices, Resumes, Legal Contracts, Custom).
   - Source citations and confidence score tracking per field.

3. **🔍 RAG & Retrieval Studio**:
   - Automatic chunking and indexing of all parsed documents.
   - Search across selected or all indexed documents.
   - Hybrid semantic & BM25 retrieval with relevance ranking.
   - Grounded Q&A generation with side-by-side source chunk citation cards.
   - Supports Local Extractive RAG, OpenAI (GPT-4o-mini), and Google Gemini.

4. **📁 Documents & Extractions Library**:
   - Browse all parsed documents and structured extractions.
   - Preview full markdown/text in modal.
   - One-click downloads for `.md`, `.txt`, and `.json`.

5. **⚙️ Settings & Key Configuration**:
   - Manage LlamaCloud API Key, OpenAI API Key, and Gemini API Key directly in the UI.

---

## 🛠️ Project Structure

```
llamaParse/
├── .env                       # Environment configuration & API keys
├── .env.example               # Example template for environment variables
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── config.py                  # Application settings & directory setup
├── app.py                     # Flask application & HTTP/API routes
├── services/
│   ├── llama_service.py       # LlamaCloud v2 client (parse & structured extract)
│   └── rag_service.py         # LlamaIndex chunking, retrieval & RAG Q&A engine
├── templates/                 # Jinja2 HTML templates
│   ├── base.html              # Core layout, responsive navigation & assets
│   ├── index.html             # Dashboard & quick parse
│   ├── parse.html             # Document parser interface
│   ├── extract.html           # Schema-based structured extraction
│   ├── rag.html               # RAG Q&A & semantic retrieval
│   ├── documents.html         # Documents library & management
│   └── settings.html          # API keys & configuration
├── static/
│   ├── css/custom.css         # Modern dark theme styles
│   └── js/main.js             # Client-side AJAX & UI handlers
└── data/                      # Local storage
    ├── uploads/               # Raw uploaded files
    ├── parsed/                # Output .md, .txt, and metadata
    └── extracted/             # Output .json and metadata
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (.env)
Copy `.env.example` to `.env` (already configured in workspace) and ensure your `LLAMA_CLOUD_API_KEY` is set:
```bash
# Optional template copy if setting up in a new environment
cp .env.example .env
```

### 3. Run the Flask Application
```bash
python app.py
```

### 2. Open in Browser
Navigate to:
```
http://localhost:5000
```

---

## ⚡ API Quick Reference

- `POST /parse` - Upload and parse document to Markdown & Text.
- `POST /extract` - Upload and extract structured JSON with schema.
- `POST /api/query` - Query RAG with prompt, doc filter, top_k, and provider:
  ```json
  {
    "prompt": "What is the total billed amount?",
    "top_k": 4,
    "provider": "auto"
  }
  ```
- `GET /download/<doc_id>/<file_type>` - Download `md`, `txt`, or `json`.