# AI Medical Data Platform -- Project Report

**Author:** Ngoga Musagi
**Date:** February 2026
**Development Environment:** Python 3.13.3, Windows 11, Docker Desktop
**GitHub:** [git@github.com:Ngoga-Musagi/AI-Medical-Platform.git](https://github.com/Ngoga-Musagi/AI-Medical-Platform)

---

## 1. Introduction

I built an AI-powered clinical decision support system that analyzes free-text clinical notes against medical guidelines. The system uses Large Language Models (LLMs) for semantic understanding of unstructured medical text, a Neo4j knowledge graph for storing clinical guidelines, and a deterministic compliance engine for scoring treatment adherence. An interactive Dash dashboard provides analytics and an integrated Medical Advisor chatbot.

The entire platform is containerized with Docker and supports multiple LLM providers -- from local models (Ollama Mistral, running inside Docker) to cloud APIs (Gemini, Claude, OpenAI).

---

## 2. System Overview

![System Overview](diagram-1.png)

The system processes 50 synthetic clinical notes through a 4-step pipeline. Each clinical note is free-form plain text (e.g., *"45 year old male with fever and productive cough for five days. Diagnosed with pneumonia. Started on amoxicillin."*), requiring semantic understanding at every stage.

---

## 3. Architecture

![Architecture](Architecture-1.png)

### 3.1 The 4-Step LLM Pipeline

Each clinical note requires **~4 LLM calls**, making this a deeply semantic pipeline:

**Step 1 -- Entity Extraction** (`src/extraction/predictor.py`):
The LLM extracts structured entities (age, sex, symptoms, diagnosis, medications) from free-text clinical notes using few-shot prompting. This is the foundation -- all downstream steps depend on accurate extraction.

**Step 2 -- Guideline Matching** (`src/guideline_engine/evaluator.py`):
The extracted diagnosis is semantically matched to the closest disease in the Neo4j knowledge graph. The LLM handles synonyms (e.g., "heart attack" matches "myocardial infarction"), abbreviations ("UTI" matches "urinary tract infection"), and partial matches ("type 2 diabetes" matches "diabetes mellitus").

**Step 3 -- Compliance Assessment** (`src/compliance/engine.py`):
The compliance engine uses a deterministic weighted formula with LLM-based semantic drug matching. The LLM matches prescribed medications against guideline drugs (e.g., "Tylenol" matches "paracetamol"), while the scoring itself is deterministic and transparent.

**Step 4 -- Explanation Generation** (`src/explainability/explainer.py`):
The LLM generates a human-readable rationale explaining the diagnosis evidence, medication assessment, and recommendations with phrase-level importance attribution.

### 3.2 Service Architecture

The system runs as 5 Docker services orchestrated by Docker Compose:

| Service | Container | Port | Role |
|---------|-----------|------|------|
| **Ollama** | medical-ollama | 11434 | LLM inference engine (Mistral/Llama3) |
| **Ollama-pull** | medical-ollama-pull | - | One-shot init container to download model |
| **Neo4j** | medical-neo4j | 7474, 7687 | Knowledge graph database |
| **API** | medical-api | 8000 | FastAPI REST API + chatbot backend |
| **Dashboard** | medical-dashboard | 8050 | Dash analytics + chat interface |

### 3.3 Key Design Decisions

- **Ollama in Docker**: Instead of requiring local Ollama installation, the LLM engine runs inside Docker. Models are stored in a persistent Docker volume, downloaded once and reused across restarts. This saves host disk space and simplifies setup.
- **Shared LLM Client**: A single `LLMClient` class (`src/llm_client.py`) abstracts all LLM providers. All modules (extractor, evaluator, compliance, explainer, chatbot) use this shared client, enabling runtime provider switching.
- **Deterministic Scoring**: While the LLM provides semantic understanding (entity extraction, drug matching), the compliance scoring itself is deterministic and transparent -- not a black-box LLM judgment.

---

## 4. Neo4j Knowledge Graph

I used Neo4j to store 19 clinical disease guidelines as a connected graph, enabling structured reasoning about treatments.

### 4.1 Graph Schema

The graph has three node types and three relationship types:

```
(:Disease {name})
    -[:RECOMMENDED_DRUG]->     (:Drug {name})
    -[:CONTRAINDICATED_DRUG]-> (:Drug {name})
    -[:REQUIRES_TEST]->        (:Test {name})
```

### 4.2 How I Built the Knowledge Graph

On API startup, `evaluator.py` loads `outputs/guidelines.json` (which contains all 19 disease guidelines) and populates Neo4j:

1. **Disease Nodes**: Each disease (e.g., "pneumonia", "myocardial infarction") becomes a `Disease` node.
2. **Drug Nodes**: Each recommended or contraindicated drug becomes a `Drug` node, connected via `RECOMMENDED_DRUG` or `CONTRAINDICATED_DRUG` relationships.
3. **Test Nodes**: Required diagnostic tests become `Test` nodes connected via `REQUIRES_TEST` relationships.
4. **MERGE Pattern**: I use Cypher `MERGE` statements to avoid duplicates -- if a drug like "aspirin" is recommended for multiple diseases, only one Drug node exists with multiple relationships.

### 4.3 Why Neo4j Helps

- **Contextual Retrieval**: When the LLM identifies "pneumonia" from a clinical note, a single Cypher query retrieves all recommended drugs, contraindicated drugs, and required tests -- providing full clinical context for compliance scoring.
- **Relationship-First**: The graph structure naturally represents medical relationships (drug-disease interactions, test requirements) that would be awkward in a relational database.
- **Semantic Bridge**: Neo4j bridges the gap between unstructured LLM output and structured clinical rules. The LLM handles fuzzy matching; Neo4j provides deterministic guideline data.

### 4.4 Graph Statistics

- 19 Disease nodes
- 19+ Drug nodes
- 15+ Test nodes
- 40+ relationships (RECOMMENDED_DRUG, CONTRAINDICATED_DRUG, REQUIRES_TEST)

---

## 5. Compliance Scoring

The compliance engine produces a score between 0.0 and 1.0 using a deterministic weighted formula:

```
Overall Score = (0.40 x Medication Score) + (0.35 x Contraindication Score) + (0.25 x Test Score)
```

### 5.1 Component Breakdown

**Medication Compliance (40% weight):**
- Calculates the ratio of guideline-recommended drugs that were actually prescribed
- Uses LLM semantic matching for drug names (e.g., "Tylenol" = "paracetamol")
- Formula: `score = matches / max(recommended_count, prescribed_count)`

**Contraindication Check (35% weight):**
- Binary score: 1.0 if no contraindicated drugs found, 0.0 if any found
- This is the most safety-critical component -- any contraindication triggers a CRITICAL alert
- Example: Ciprofloxacin for UTI is contraindicated per guidelines; prescribing it drops this score to 0.0

**Test Completeness (25% weight):**
- Checks if required diagnostic tests were mentioned or ordered
- Since clinical notes rarely mention test orders explicitly, a partial credit of 0.5 is given when required tests exist
- Score is 1.0 when no tests are required by guidelines

### 5.2 Dashboard Display

| Overall Score | Display Status | Meaning |
|--------------|----------------|---------|
| >= 0.80 | Compliant (Green) | Treatment follows guidelines |
| 0.50 - 0.79 | Partially Compliant (Yellow) | Some guideline deviations |
| < 0.50 | Non-Compliant (Red) | Significant guideline deviations |
| Any contraindication | CRITICAL (Red + Alert) | Safety concern -- immediate review |

---

## 6. Medical Advisor Chatbot

I built a conversational AI agent (`src/chatbot/agent.py`) that helps doctors get guideline-grounded recommendations.

### 6.1 Agent Architecture

1. **Guidelines as Context**: All 19 disease guidelines are formatted and injected into the system prompt, giving the LLM a complete clinical knowledge base.
2. **Structured System Prompt**: The prompt includes rules for when to flag contraindications, recommend tests, cite guidelines, and remind clinicians that AI recommendations require clinical judgment.
3. **Conversation Memory**: Each session stores conversation history (last 10 exchanges), enabling follow-up questions like "what about the test results?" in context.
4. **Dynamic Guideline Detection**: After each response, the agent scans the conversation to detect which guidelines were referenced, returning structured metadata (matched diseases, recommended drugs, required tests) alongside the response text.
5. **Provider Agnostic**: Works with any configured LLM through the shared `LLMClient`.

---

## 7. Dashboard

### 7.1 Analytics Tab

![Dashboard Analytics](Dashboard-1.png)

The analytics dashboard provides:
- **Summary Cards**: Total notes, average compliance score, contraindication alerts, missing tests (clickable to filter)
- **Status Filters**: Filter by Compliant, Partially Compliant, CRITICAL, Non-Compliant
- **Charts**: Compliance distribution histogram, compliance by disease bar chart, test completion stacked bars, alerts by disease
- **Interactive Table**: Color-coded rows, sortable, filterable, click any row for full compliance detail
- **LLM Provider Selector**: Switch providers from the navbar without restarting

### 7.2 Chat Tab

![Dashboard Chat](Dashboard-2.png)

The chat interface provides:
- Multi-line input for clinical scenario descriptions
- Dynamic guideline tags showing which guidelines each response references
- Session-based conversation memory
- Guidelines sidebar with all 19 diseases
- Quick prompt buttons for common scenarios

---

## 8. LLM Provider Comparison

The system supports 6 LLM providers. Each note requires ~4 LLM calls.

### 8.1 Hypothetical Inference Comparison

| Provider | Type | Speed per Note | Cost | Privacy |
|----------|------|---------------|------|---------|
| Mistral 7B (Docker Ollama) | Local CPU | ~2-7 min | Free | Full |
| Llama 3 8B (Docker Ollama) | Local CPU | ~3-8 min | Free | Full |
| Gemini 2.0 Flash | Cloud API | ~15-30s | Free tier | Data sent externally |
| Claude Sonnet | Cloud API | ~15-30s | Paid | Data sent externally |
| GPT-4 Turbo | Cloud API | ~15-30s | Paid | Data sent externally |

Each note requires ~4 sequential LLM calls, so total time per note is cumulative. Local models run on CPU inference inside Docker, which is significantly slower than cloud-hosted GPU inference. Cloud APIs run on optimized GPU clusters. The trade-off is privacy vs. speed.

**GPU Acceleration & Data Privacy:** With a local GPU (e.g., NVIDIA with CUDA support), local inference times would drop dramatically -- potentially matching cloud API speeds while keeping all patient data on-premises. This is the ideal setup for the medical industry, where data privacy and regulatory compliance (HIPAA, GDPR) are critical. Running models locally ensures no sensitive clinical data leaves the organization's infrastructure, eliminating third-party data exposure risks entirely.

### 8.2 Note on Meditron

Meditron 7B is technically functional -- it connects and returns responses through the Ollama Docker service. However, the output quality is limited because Meditron is a **base pretrained model** (not instruction-tuned like Mistral or Llama 3). This means it struggles with structured JSON output, which our pipeline relies on for entity extraction and semantic matching. The provider connection is fully functional, but for production use, instruction-tuned models (Mistral, Llama 3, or cloud APIs) are recommended for reliable structured output.

### 8.3 API Key Requirements

- **Ollama (default)**: No key needed, runs in Docker
- **Gemini**: Free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Claude**: Paid API key from [Anthropic Console](https://console.anthropic.com/)
- **OpenAI**: Paid API key from [OpenAI Platform](https://platform.openai.com/api-keys)

---

## 9. How to Run

### Prerequisites
- Docker Desktop (includes Docker Compose V2)
- Git Bash (Windows) or any bash shell (Linux/macOS)

### Steps

```bash
# 1. Clone
git clone git@github.com:Ngoga-Musagi/AI-Medical-Platform.git
cd AI-Medical-Platform

# 2. Configure
cp .env.example .env
# Edit .env to add API keys if using cloud providers

# 3. Start
bash run.sh                  # Default: Ollama Mistral (free, private)
bash run.sh gemini           # Or use Gemini (cloud, free tier)

# 4. Access
# Dashboard:  http://localhost:8050
# API Docs:   http://localhost:8000/docs
# Neo4j:      http://localhost:7474

# 5. Batch analyze all 50 notes
bash run.sh batch

# 6. Stop
bash run.sh stop
```

See `.env.example` for all configuration options.

---

## 10. Project Structure

```
AI-Medical-Platform/
├── README.md                  # Full documentation
├── Report.md                  # This report
├── Dockerfile                 # Container definition
├── docker-compose.yml         # 5 services
├── requirements.txt           # Python dependencies
├── run.sh                     # Cross-platform startup
├── run_batch.py               # Batch analysis script
├── .env.example               # Configuration template
│
├── src/
│   ├── config.py              # Centralized configuration
│   ├── llm_client.py          # Shared LLM client
│   ├── extraction/predictor.py       # Entity extraction
│   ├── guideline_engine/evaluator.py # Neo4j + guideline matching
│   ├── compliance/engine.py          # Compliance scoring
│   ├── explainability/explainer.py   # Explanation generation
│   ├── chatbot/agent.py              # Medical Advisor agent
│   ├── api/app.py                    # FastAPI REST API
│   ├── dashboard/app.py              # Dash dashboard
│   ├── evaluation/evaluate.py        # Evaluation metrics
│   └── mlops/tracker.py              # Prediction logging
│
└── outputs/
    ├── guidelines.json        # 19 clinical guidelines
    ├── clinical_notes.json    # 50 synthetic notes
    ├── ground_truth.json      # Ground truth (15 notes)
    └── sample_predictions.json
```

---

## 11. Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.13.3 | Development language |
| FastAPI | REST API framework |
| Dash (Plotly) | Interactive analytics dashboard |
| Neo4j 5.15 | Knowledge graph database |
| Docker + Docker Compose | Container orchestration |
| Ollama | Local LLM inference engine |
| Mistral 7B / Llama 3 8B | Default local LLM models |
| Google Gemini / Claude / GPT-4 | Cloud LLM alternatives |

---

*End of Report*
