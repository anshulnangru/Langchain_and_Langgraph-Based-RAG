#  RAG Assistant — Production Agentic RAG Pipeline

> A stateful **Agentic RAG system** built with LangGraph and FastAPI: intelligent query routing, cross-encoder reranking, lightweight guardrails, multi-key LLM gateway, and real-time token streaming to a React chat UI.

**Live demo:** https://langchain-and-langgraph-based-rag.vercel.app/

---

## Architecture Overview

```
User Query (React UI)
    │  POST /stream  (x-api-key, per-IP rate limit)
    ▼
┌─────────────────────────────┐
│   Guardrails Gate           │  ← regex rules (jailbreak, identity, greeting,
│                             │     farewell, capabilities) + Groq classifier
└────────────┬────────────────┘     for unclear messages
             │ (clean query)
             ▼
┌─────────────────────────────┐
│     LangGraph Agent         │
│  ┌────────────────────────┐ │
│  │   Planner Node         │ │  ← Qwen (Groq) decides CONVERSATIONAL
│  │   (intent routing)     │ │     or writes a standalone search query
│  └──────────┬─────────────┘ │
│    ┌────────┴────────┐      │
│    ▼                 ▼      │
│  CONVERSATIONAL   TECHNICAL │
│  (memory only)   (retrieval)│
│                      │      │
│  ┌───────────────────▼───┐  │
│  │   Retriever Node      │  │  ← Jina embedding → Qdrant (top 15)
│  │   + FlashRank Rerank  │  │     → cross-encoder rerank → top 7
│  └───────────┬───────────┘  │
│  ┌───────────▼───────────┐  │
│  │   Responder Node      │  │  ← gpt-oss-20b (Groq) synthesizes answer
│  └───────────────────────┘  │
└─────────────────────────────┘
             │
             ▼
    SSE Token Streaming → React UI
```

---

## Key Features

| Feature | Implementation |
|---|---|
| **Agentic Routing** | LangGraph planner classifies queries as conversational or technical, and rewrites follow-ups ("show an example of that") into standalone search queries |
| **Semantic Reranking** | FlashRank `ms-marco-MiniLM-L-12-v2` cross-encoder reranks the top-15 Qdrant results; a per-source cap keeps the final top-7 diverse |
| **Jina Embeddings** | `jina-embeddings-v3` (1024-dim); startup fails fast if the embedding dimension is wrong |
| **Guardrails** | Lightweight regex rules plus a Groq classifier for off-topic messages (fails open if the classifier is unavailable) |
| **LLM Gateway** | Groq via Portkey (fallback + cache) or direct Groq, rotating across up to 4 API keys on rate limits / 5xx |
| **Conversation Memory** | LangGraph `MemorySaver` keyed by `thread_id`; prompts use the last 3 turns |
| **SSE Streaming** | Real token streaming from the responder node; `<think>` blocks are hidden |
| **API Protection** | `x-api-key` auth, per-IP and global rate limits, CORS allow-list |
| **Chat History** | Sidebar of previous chats saved in the browser's localStorage |
| **Evaluation** | LLM-as-judge scripts for faithfulness, relevancy, contextual recall and freshness, logged to LangSmith |
| **Observability** | Logfire spans around every pipeline stage (exported when `logfire.configure()` runs — currently only in the ingestion job) |
| **Noisy Corpus** | ~36% true LangGraph/LangChain docs, ~64% semantically similar noise (LlamaIndex, Haystack) to stress-test retrieval |

---

##  Tech Stack

**AI/ML**
- [LangGraph](https://github.com/langchain-ai/langgraph) / LangChain — stateful agent orchestration
- [Groq](https://groq.com) — `openai/gpt-oss-20b` (answers) and `qwen/qwen3.8-27b` (planner)
- [Portkey](https://portkey.ai) — optional LLM gateway (fallback, caching)
- [Jina AI](https://jina.ai) — `jina-embeddings-v3` embeddings (1024-dim)
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) — cross-encoder reranking

**Infrastructure**
- [Qdrant Cloud](https://qdrant.tech) — vector database (cosine similarity)
- [FastAPI](https://fastapi.tiangolo.com) — async API with SSE streaming, SlowAPI rate limiting
- Docker — backend image with the reranker model baked in
- [Logfire](https://logfire.pydantic.dev) — observability; [LangSmith](https://smith.langchain.com) — eval tracking

**Frontend**
- React + Vite — chat UI with markdown rendering and chat-history sidebar (deployed on Vercel)

---

## Project Structure

```
RAG/
├── main.py                    # FastAPI app — lifespan checks, /stream, /query, /graph, /health
├── Dockerfile                 # Backend image (python:3.12-slim, bakes FlashRank model)
├── requirements-prod.txt      # Runtime deps used by the Docker image
├── src/
│   ├── agents/
│   │   ├── graph.py           # LangGraph StateGraph (planner → retriever → responder)
│   │   ├── state.py           # AgentState TypedDict
│   │   └── nodes/
│   │       ├── planner.py     # Intent routing + query rewriting
│   │       ├── retriever.py   # Calls the retrieval service
│   │       └── responder.py   # Answer synthesis
│   ├── retrieval/
│   │   ├── embeddings.py      # Jina embeddings (query + passage tasks)
│   │   ├── qdrant_service.py  # Qdrant search, retry, rerank, source-diversity cap
│   │   └── ranking_service.py # FlashRank cross-encoder
│   ├── ingestion/
│   │   ├── processor.py       # Ingestion pipeline: load → clean → chunk → embed → upsert
│   │   ├── text_cleanup.py    # Strips markup noise
│   │   ├── chunking/          # Text splitter
│   │   └── loaders/           # HTML, PDF, Office, text
│   ├── guardrails/
│   │   └── rails.py           # Regex rules + Groq classifier
│   ├── gateways/
│   │   └── client.py          # LLM client: Portkey or direct Groq, key rotation
│   └── config/
│       └── config.py          # Centralised settings from .env
├── evals/                     # freeze_rag.py + LLM-judge metric scripts
├── data/
│   ├── true/                  # LangGraph + LangChain docs
│   └── noisy/                 # LlamaIndex + Haystack docs
└── ui/                        # React + Vite frontend
```

---

## Setup & Running

### Prerequisites
- Python 3.12+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Clone & install

```bash
git clone https://github.com/anshulnangru/Langchain_and_Langgraph-Based-RAG
cd Langchain_and_Langgraph-Based-RAG
uv sync
```

### 2. Environment variables

Create a `.env` file in the project root:

```env
# API protection (required — the server refuses to start without it)
APP_API_KEY=choose_a_secret_key
CORS_ORIGINS=http://localhost:5173
RATE_LIMIT=10/minute
GLOBAL_RATE_LIMIT=100/hour

# Groq (LLM inference) — up to four keys, rotated on rate limits
GROQ_API_KEY_1=your_groq_key
GROQ_API_KEY_2=
GROQ_API_KEY_3=
GROQ_API_KEY_4=
GROQ_MODEL=openai/gpt-oss-20b
GROQ_MODEL_PLANNER=qwen/qwen3.8-27b

# Portkey gateway (optional)
USE_PORTKEY=false
PORTKEY_API_KEY=
GROQ_SLUG=

# Qdrant (vector database)
QDRANT_CLUSTER_ENDPOINT=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key

# Jina (embeddings)
JINA_API_KEY=your_jina_api_key

# Observability (optional; Logfire is used by the ingestion job)
LOGFIRE_TOKEN=your_logfire_token
LANGSMITH_API_KEY=your_langsmith_key
```

### 3. Ingest documents

```bash
# Index every subfolder of data/ into Qdrant (wipes the collection first)
python -m src.ingestion.processor data --wipe
```

### 4. Start the backend

```bash
uvicorn main:app --reload --port 8000
```

On startup the app checks `APP_API_KEY`, verifies Jina returns 1024-dim embeddings, loads the reranker and the guardrails, then logs:
```
App startup complete - guardrails initialised.
```

Or run it with Docker:

```bash
docker build -t agentic-rag .
docker run --env-file .env -e PORT=8000 -p 8000:8000 agentic-rag
```

### 5. Start the frontend

```bash
cd ui
echo "VITE_API_URL=http://localhost:8000" > .env
echo "VITE_API_KEY_NEW=choose_a_secret_key" >> .env
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

> Note: `VITE_*` variables are compiled into the public JS bundle, so the frontend key is not a real secret. The rate limits are what protect the API.

---

##  How the Pipeline Works

### Query Routing

Every message passes through two gates before retrieval:

1. **Guardrails** — fast regex rules answer greetings, farewells and capability questions, and block jailbreak attempts and identity probing with canned responses. Messages that match no rule and aren't obviously on-topic go to a small Groq classifier that flags off-topic requests. Blocked messages never reach the LangGraph pipeline.

2. **LangGraph Planner** — for messages that pass, the planner decides whether the message is `CONVERSATIONAL` (greeting, thanks, questions about the user or the chat; answered from memory) or technical. For technical messages it outputs a standalone search query, resolving follow-ups from the conversation history.

### Retrieval & Reranking

Technical queries trigger a two-stage retrieval:

1. **Vector search** — Qdrant returns the top 15 candidate chunks by cosine similarity over 1024-dim Jina embeddings (with a retry on transient connection errors).
2. **Cross-encoder reranking** — FlashRank's `ms-marco-MiniLM-L-12-v2` scores each candidate against the query. A per-source cap (default 2 chunks per document) keeps the final top 7 diverse.

Cosine similarity is fast but fuzzy, so it finds candidates; the cross-encoder is slower but precise, so it picks the winners. If retrieval returns nothing, the responder says so instead of answering without context.

### Noisy Corpus Design

The corpus deliberately includes ~64% noise (semantically related but wrong-framework documentation from LlamaIndex and Haystack). This stress-tests the retrieval pipeline's ability to surface true LangGraph/LangChain content, mirroring real deployments where only a fraction of indexed data is relevant to a given query.

---

##  Evaluation

```bash
# 1. Run the RAG pipeline over the golden questions and freeze the outputs
python evals/freeze_rag.py

# 2. Score the frozen outputs with an LLM judge (each logs to LangSmith)
python evals/eval_faithfulness.py
python evals/eval_relevancy.py
python evals/eval_contextual_recall.py
python evals/eval_freshness.py
```

Each metric script accepts `--input`, `--output` and `--model`.

---

##  Security

- **API key** — every `/stream`, `/query` and `/graph` request needs a matching `x-api-key` header.
- **Rate limiting** — per-IP and global limits via SlowAPI.
- **Jailbreak / prompt injection** — persona reassignment and "ignore your instructions" patterns are blocked.
- **Identity probing** — "what model are you / are you ChatGPT" questions get a neutral response.
- **Off-topic queries** — non-AI-engineering questions are redirected.

---

##  API Reference

### `POST /stream`
Streams the response token by token via Server-Sent Events. Requires `x-api-key`.

```json
// Request
{ "q": "How does MemorySaver work in LangGraph?", "thread_id": "user_123" }

// SSE stream
data: {"token": "Memory"}
data: {"token": "Saver"}
...
data: [DONE]
```

### `POST /query`
Returns the full response in one JSON payload (no streaming). Requires `x-api-key`.

### `GET /graph`
Returns a Mermaid PNG diagram of the LangGraph agent workflow. Requires `x-api-key`.

### `GET /health`
Cheap liveness check, returns `{"status": "ok"}`.

---

##  Roadmap

- [x] **Eval pipeline** — LLM-as-judge faithfulness, relevancy, contextual recall and freshness on a golden Q&A set
- [x] **Portkey gateway** — LLM fallback routing + caching
- [x] **Public demo** — Vercel frontend + Dockerized backend
- [ ] **AWS IaC** — Terraform for ECS Fargate + ALB + Secrets Manager
- [ ] **Fine-tuning** — LoRA/QLoRA on a domain-specific task

---

##  Author

**Anshul Nangru** · [GitHub](https://github.com/anshulnangru) · [LinkedIn](https://www.linkedin.com/in/anshul-nangru/)

---

*Built as a portfolio project demonstrating production-grade agentic AI engineering.*
