<p align="center">
  <h1 align="center">🧠 Enterprise Smart Knowledge-Base</h1>
  <p align="center">
    AI-powered document assistant with Hybrid Search (Semantic + Keyword) and built-in hallucination detection
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
    <img src="https://img.shields.io/badge/Pinecone-Vector%20DB-00C896?style=for-the-badge" alt="Pinecone">
    <img src="https://img.shields.io/badge/LangChain-0.1.13-1C3C3C?style=for-the-badge" alt="LangChain">
  </p>
</p>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Core Technologies](#-core-technologies)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Environment Configuration](#-environment-configuration)
- [Installation \& Setup](#-installation--setup)
- [Development Server](#-development-server)
- [Docker Configuration](#-docker-configuration)
- [Running Tests](#-running-tests)
- [Evaluation Results](#-evaluation-results)
- [Configuration Reference](#-configuration-reference)
- [Troubleshooting](#-troubleshooting)
- [Resource Links](#-resource-links)
- [License](#-license)

---

## 🔍 Project Overview

**Enterprise Smart Knowledge-Base** is a Retrieval-Augmented Generation (RAG) application built with Streamlit. It enables users to upload PDF documents, automatically ingest and chunk them into a searchable knowledge base, and ask questions in natural language. The system combines **semantic vector search** (via Pinecone) with **keyword-based BM25 retrieval** using Reciprocal Rank Fusion (RRF) to deliver highly relevant answers.

Every response is grounded in the uploaded documents, and a dedicated **Fact-Check Judge** (a separate LLM evaluator) automatically verifies that answers are faithful to the source material — flagging any potential hallucinations with a clear PASS/FAIL rating.

The application is designed with **Thai language support** at its core, including Thai text tokenization (via PyThaiNLP), broken-vowel correction, and Thai-English boundary normalization.

---

## ✨ Key Features

| Feature                     | Description                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **PDF Ingestion**           | Upload and process PDF documents with automatic text extraction and cleaning                                            |
| **Thai NLP Pipeline**       | Full Thai language preprocessing — broken vowel fixes, word tokenization (`newmm`), and Thai-English boundary detection |
| **Hybrid Search**           | Combines Pinecone vector similarity search with BM25 keyword retrieval using Reciprocal Rank Fusion (RRF)               |
| **Conversational AI**       | Chat interface powered by OpenTyphoon LLM (`typhoon-v2.5-30b-a3b-instruct`) with conversation history context           |
| **Faithfulness Evaluation** | Automated Fact-Check Judge that verifies every answer against source documents (PASS / FAIL)                            |
| **Performance Metrics**     | Real-time display of response time and evaluation results                                                               |
| **Source Attribution**      | Expandable source context panel showing exact referenced passages and page numbers                                      |
| **Docker Support**          | Fully containerized with Docker and Docker Compose for consistent deployment                                            |

---

## 🛠️ Core Technologies

| Category              | Technology                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------ |
| **Frontend / UI**     | [Streamlit](https://streamlit.io/) `1.32.0`                                                |
| **LLM Provider**      | [OpenTyphoon AI](https://opentyphoon.ai/) (`typhoon-v2.5-30b-a3b-instruct`, `typhoon-ocr`) |
| **Orchestration**     | [LangChain](https://www.langchain.com/) `0.1.13`                                           |
| **Vector Database**   | [Pinecone](https://www.pinecone.io/) `3.2.2`                                               |
| **Embeddings**        | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) via Sentence Transformers                |
| **Keyword Retrieval** | [rank-bm25](https://github.com/dorianbrown/rank_bm25) `0.2.2`                              |
| **Thai NLP**          | [PyThaiNLP](https://pythainlp.github.io/)                                                  |
| **PDF Parsing**       | [pypdf](https://github.com/py-pdf/pypdf) `4.1.0`                                           |
| **ML Framework**      | [PyTorch](https://pytorch.org/) (CPU)                                                      |
| **Containerization**  | [Docker](https://www.docker.com/) + Docker Compose                                         |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                   │
│              Chat Interface  ·  Sidebar  ·  Metrics         │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌─────────────┐ ┌──────────┐ ┌─────────────┐
   │  Ingestion  │ │   RAG    │ │  Evaluator  │
   │   Pipeline  │ │  Engine  │ │   (Judge)   │
   └──────┬──────┘ └────┬─────┘ └──────┬──────┘
          │             │              │
          ▼             ▼              ▼
   ┌─────────────┐ ┌──────────┐ ┌─────────────┐
   │ PDF Loader  │ │  Hybrid  │ │  Typhoon    │
   │ + Chunker   │ │  Search  │ │  Instruct   │
   │ (ThaiNLP)   │ │  (RRF)   │ │    LLM      │
   └─────────────┘ └────┬─────┘ └─────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       ┌─────────────┐     ┌──────────────┐
       │   Pinecone  │     │  Local BM25  │
       │  (Vectors)  │     │  (Pickle)    │
       └─────────────┘     └──────────────┘
```

**Query Flow:**

1. User uploads a PDF → text is extracted, cleaned, tokenized, and chunked
2. Chunks are embedded with `BAAI/bge-m3` → stored in Pinecone (vectors) + local BM25 index (keywords)
3. User asks a question → Hybrid Search (Semantic + BM25) retrieves top-k documents via RRF
4. Retrieved context + chat history → OpenTyphoon LLM generates a grounded answer
5. Fact-Check Judge independently verifies faithfulness → PASS or FAIL

---

## 📁 Project Structure

```
rag-streamlit/
├── app.py                      # Main Streamlit application entry point
├── config.yaml                 # Application configuration (chunking, embedding, vector DB)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container image definition
├── docker-compose.yml          # Multi-service Docker orchestration
├── .env                        # Environment variables (API keys)
├── .dockerignore               # Docker build exclusions
├── eval_dataset.json           # Evaluation dataset (37 questions, 3 difficulty levels)
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── ingestion/              # Document ingestion pipeline
│   │   ├── __init__.py
│   │   ├── ingestion.py        #   PDF loading, text cleaning, and chunking pipeline
│   │   └── pdf_processor.py    #   Smart PDF processing pipeline
│   │
│   ├── rag/                    # Retrieval-Augmented Generation core
│   │   ├── __init__.py
│   │   ├── retrieval.py        #   Hybrid store, embeddings, and RRF search
│   │   ├── generator.py        #   LLM answer generation (OpenTyphoon)
│   │   └── evaluator.py        #   Faithfulness & accuracy evaluator
│   │
│   ├── ui/                     # Streamlit UI components
│   │   ├── __init__.py
│   │   ├── chat.py             #   Chat history, source display, evaluation metrics
│   │   └── sidebar.py          #   Sidebar: file upload & system status
│   │
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       ├── helpers.py          #   Markdown cleaning, time formatting
│       └── logger.py           #   Dual-output logger (console + file)
│
├── scripts/                    # Evaluation & benchmarking scripts
│   ├── run_eval.py             #   RAG faithfulness evaluation pipeline
│   ├── run_raw_llm.py          #   Raw LLM baseline runner (Gemini)
│   └── eval_raw_llm.py         #   Raw LLM accuracy evaluator
│
├── tests/                      # Test suite
│   └── test_loader.py          #   Unit tests for chunking logic
│
├── local_bm25_data/            # Persisted BM25 index (bm25_index.pkl)
├── vector_db/                  # Local vector DB data (if applicable)
└── logs/                       # Runtime logs (system.log, auto-created)
```

---

## 📋 Prerequisites

- **Python** 3.10+
- **Docker** & **Docker Compose** (for containerized deployment)
- **Pinecone** account with an index named `rag-streamlit`
- **OpenTyphoon AI** API key

---

## 🔐 Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
TYPHOON_API_KEY=your_typhoon_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

| Variable           | Description                                                                                   |
|--------------------|-----------------------------------------------------------------------------------------------|
| `TYPHOON_API_KEY`  | API key for OpenTyphoon AI — used for LLM response generation and fact-check evaluation       |
| `PINECONE_API_KEY` | API key for Pinecone — used for vector storage and semantic retrieval                         |

> **⚠️ Important:** Never commit your `.env` file to version control. It is excluded via `.dockerignore` and should be added to `.gitignore`.

---

## 🚀 Installation & Setup

### Local Development

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd rag-streamlit
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   # venv\Scripts\activate         # Windows
   ```

3. **Install PyTorch (CPU)**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

6. **Set up Pinecone**

   Create an index named `rag-streamlit` in your [Pinecone dashboard](https://app.pinecone.io/). The embedding dimension should match `BAAI/bge-m3` output (1024 dimensions).

---

## 🖥️ Development Server

Start the Streamlit application locally:

```bash
streamlit run app.py
```

The app will be available at **http://localhost:8501**.

### Usage

1. Open the app in your browser
2. Upload a PDF document via the sidebar
3. Wait for the ingestion pipeline to complete (status shown in sidebar)
4. Start asking questions in the chat interface
5. View source references and fact-check results below each answer

---

## 🐳 Docker Configuration

### Dockerfile

The project uses a multi-stage build based on `python:3.10-slim`:

- Installs system build dependencies (`build-essential`, `gcc`, `python3-dev`)
- Pre-installs PyTorch CPU to prevent memory issues during the build
- Copies and installs Python requirements
- Creates necessary data directories
- Exposes port `8501` with a health check endpoint

### Docker Compose

```yaml
services:
  rag-streamlit:
    build: .
    container_name: rag-streamlit
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./local_bm25_data:/app/local_bm25_data
      - ./vector_db:/app/vector_db
      - ./uploaded_docs:/app/uploaded_docs
    restart: unless-stopped
```

**Volumes** persist the BM25 index, vector DB data, and uploaded PDF files between container restarts.

### Run with Docker Compose

```bash
# Build and start
docker compose up --build -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

The containerized app will be accessible at **http://localhost:8501**.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

The test suite currently includes:

- **`test_ingestion.py`** — Validates the document chunking pipeline produces multiple chunks with preserved metadata

---

## ⚙️ Configuration Reference

Application behavior is controlled via `config.yaml`:

```yaml
ingestion:
  chunk_size: 800 # Maximum characters per chunk
  chunk_overlap: 150 # Overlap between consecutive chunks

embedding:
  model_name: "BAAI/bge-m3" # HuggingFace embedding model
  device: "cpu" # Inference device (cpu / cuda)

vector_db:
  persist_directory: "./local_bm25_data" # Local BM25 index storage
  index_name: "rag-streamlit" # Pinecone index name
```

---

## 📊 Evaluation Results

### RAG Performance

| Difficulty | Faithfulness | Accuracy | Avg Latency |
|-----------|-------------|----------|------------|
| EASY      | 94.0%       | 98.0%    | 6.07s      |
| MEDIUM    | 96.0%       | 94.0%    | 8.06s      |
| HARD      | 74.0%       | 84.0%    | 9.91s      |

Overall:
- Faithfulness: 88.0%
- Accuracy: 92.0%
- Avg Latency: 8.01s
- P95 Latency: 14.60s

---

### No RAG Baseline

| Difficulty | Accuracy | Avg Latency |
|-----------|----------|------------|
| EASY      | 32.0%    | 1.03s      |
| MEDIUM    | 32.0%    | 0.95s      |
| HARD      | 56.0%    | 1.85s      |

Overall:
- Accuracy: 40.0%
- Avg Latency: 1.28s

---

## 📈 Insights

- Accuracy improves from 40% → 92%
- Better grounding and factual correctness
- Latency increases due to retrieval + evaluation
- Hard queries remain challenging

---

## 🔗 Resource Links

| Resource                | URL                                      |
| ----------------------- | ---------------------------------------- |
| Streamlit Documentation | https://docs.streamlit.io/               |
| LangChain Documentation | https://python.langchain.com/docs/       |
| Pinecone Documentation  | https://docs.pinecone.io/                |
| OpenTyphoon AI          | https://opentyphoon.ai/                  |
| BAAI/bge-m3 Model Card  | https://huggingface.co/BAAI/bge-m3       |
| PyThaiNLP Documentation | https://pythainlp.github.io/             |
| rank-bm25               | https://github.com/dorianbrown/rank_bm25 |

---

## 🛠️ Troubleshooting

### Out of Memory (OOM) when Loading the Embedding Model

The `BAAI/bge-m3` embedding model is large (~2.3 GB). If the application crashes with an `OOM` error at startup:

- **Local development**: Ensure your machine has at least **4 GB of free RAM**. Close other memory-intensive applications.
- **Docker**: Increase the Docker Desktop memory limit (Settings → Resources → Memory) to at least **6 GB**.
- **Alternative**: Switch to a smaller model (e.g., `BAAI/bge-small-en`) in `config.yaml` for resource-constrained environments, noting it will reduce retrieval quality.

### Pinecone 1024 Dimension Mismatch

If you see an error like `Index dimension mismatch: expected 1024, got N` when upserting vectors:

1. The `BAAI/bge-m3` model outputs **1024-dimensional** vectors — make sure your Pinecone index was created with `dimension=1024`.
2. If you previously used a different embedding model, **delete and recreate** the index in the [Pinecone console](https://app.pinecone.io/) with the correct dimension.
3. Verify the `index_name` in `config.yaml` matches the index name in your Pinecone dashboard exactly.

### Uploaded Files Disappear After Streamlit Reruns

Streamlit clears in-memory file buffers on each rerun. This application addresses this by saving uploaded PDFs to the `uploaded_docs/` directory on disk. If files still seem to disappear:

- Confirm the `uploaded_docs/` directory exists (or that the application has write permissions to create it).
- When running via Docker, verify the `./uploaded_docs:/app/uploaded_docs` volume is present in `docker-compose.yml`.

### Pinecone API Key / Connection Errors

- Double-check that `PINECONE_API_KEY` in your `.env` file is valid and not expired.
- Ensure the Pinecone index name in `config.yaml` (`vector_db.index_name`) matches the index in your account.
- The free Pinecone tier only supports one index — delete unused indexes if you hit the limit.

---

## 🔗 Resource Links

| Resource                | URL                                      |
| ----------------------- | ---------------------------------------- |
| Streamlit Documentation | https://docs.streamlit.io/               |
| LangChain Documentation | https://python.langchain.com/docs/       |
| Pinecone Documentation  | https://docs.pinecone.io/                |
| OpenTyphoon AI          | https://opentyphoon.ai/                  |
| BAAI/bge-m3 Model Card  | https://huggingface.co/BAAI/bge-m3       |
| PyThaiNLP Documentation | https://pythainlp.github.io/             |
| rank-bm25               | https://github.com/dorianbrown/rank_bm25 |

---

## 📄 License

This project is currently unlicensed. To specify usage terms, create a `LICENSE` file in the project root. Common choices include:

- [MIT License](https://choosealicense.com/licenses/mit/) — permissive, widely used for open source
- [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) — permissive with patent protection
- [GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/) — copyleft, requires derivative works to be open source

Visit [choosealicense.com](https://choosealicense.com/) for guidance on selecting the right license for your project.
