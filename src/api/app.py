"""
FastAPI Application - AI Medical Data Platform
REST API with clinical analysis, medical chatbot, provider switching, and MLOps tracking.
Supports local (Ollama) and cloud (Gemini, Claude, OpenAI) LLM providers.
"""

import logging
import os
import time
import uuid
import sys
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

sys.path.append(str(Path(__file__).parent.parent))
from config import config, LLMProvider
from explainability.explainer import TreatmentAgent
from chatbot.agent import MedicalAdvisorAgent
from mlops.tracker import PredictionLogger, PerformanceTracker, ModelRegistry

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Global state
agent: Optional[TreatmentAgent] = None
chatbot: Optional[MedicalAdvisorAgent] = None
prediction_logger = PredictionLogger()
performance_tracker = PerformanceTracker()
model_registry = ModelRegistry()


# --- Pydantic Models ---


class ClinicalNote(BaseModel):
    note_id: str = Field(..., min_length=1, max_length=50, description="Unique note identifier")
    text: str = Field(..., min_length=10, max_length=10000, description="Clinical note text")

    @field_validator("text")
    @classmethod
    def text_must_have_content(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Clinical note must be at least 10 characters")
        return v


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="Doctor's clinical query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    matched_guidelines: List[str] = []
    guideline_details: Dict = {}
    llm_provider: str = ""
    latency_ms: float = 0.0
    history_length: int = 0


class ProviderRequest(BaseModel):
    provider: str = Field(..., description="LLM provider name (e.g. ollama-llama3, gemini, claude)")


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global agent, chatbot
    logger.info("Starting AI Medical Platform...")

    if not config.validate():
        logger.error("Configuration invalid")
        raise RuntimeError("Check .env file")

    agent = TreatmentAgent()
    agent.evaluator.load_guidelines()

    chatbot = MedicalAdvisorAgent()

    logger.info(f"System ready - Provider: {config.LLM_PROVIDER}")
    yield
    # Shutdown
    if agent:
        agent.evaluator.close()
    logger.info("System shutdown")


# --- App ---


app = FastAPI(
    title="AI Medical Data Platform",
    version="3.0",
    description=f"Clinical decision support + Medical Advisor Chatbot | LLM: {config.LLM_PROVIDER}",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Middleware ---


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {latency_ms:.0f}ms")
    return response


# --- Clinical Analysis Endpoints ---


@app.get("/")
async def root():
    return {
        "service": "AI Medical Data Platform",
        "version": "3.0",
        "llm_provider": config.LLM_PROVIDER,
        "status": "running",
        "endpoints": {
            "analyze": "POST /analyze_note",
            "chat": "POST /chat",
            "health": "GET /health",
            "providers": "GET /providers",
            "dashboard": f"http://localhost:{config.DASHBOARD_PORT}",
        },
    }


@app.get("/health")
async def health():
    neo4j_ok = False
    if agent:
        try:
            with agent.evaluator.driver.session() as session:
                session.run("RETURN 1")
            neo4j_ok = True
        except Exception:
            pass

    return {
        "status": "healthy" if (agent and neo4j_ok) else "degraded",
        "llm_provider": config.LLM_PROVIDER,
        "neo4j_connected": neo4j_ok,
        "neo4j_uri": config.NEO4J_URI,
        "model_info": model_registry.get_model_info(),
        "chatbot_ready": chatbot is not None,
    }


@app.post("/analyze_note")
async def analyze_note(note: ClinicalNote) -> Dict:
    """Analyze a clinical note: extract entities, match guidelines, score compliance."""
    if not agent:
        raise HTTPException(503, "System not initialized")

    start = time.time()
    try:
        result = agent.analyze(note.text)
        result["note_id"] = note.note_id
        result["llm_provider"] = config.LLM_PROVIDER

        latency_ms = (time.time() - start) * 1000
        performance_tracker.record(latency_ms, True)
        prediction_logger.log_prediction(note.note_id, note.text, result, latency_ms)

        return result
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        performance_tracker.record(latency_ms, False)
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.post("/analyze")
async def analyze(note: ClinicalNote) -> Dict:
    """Alias for /analyze_note."""
    return await analyze_note(note)


@app.post("/batch_analyze")
async def batch_analyze(notes: List[ClinicalNote]) -> Dict:
    """Batch analyze multiple clinical notes."""
    if not agent:
        raise HTTPException(503, "System not initialized")

    results = []
    for note in notes:
        try:
            result = agent.analyze(note.text)
            result["note_id"] = note.note_id
            result["llm_provider"] = config.LLM_PROVIDER
            results.append(result)
        except Exception as e:
            results.append({"note_id": note.note_id, "status": "error", "error": str(e)})

    return {"total": len(notes), "results": results}


# --- Medical Chatbot Endpoints ---


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Medical Advisor Chatbot: guideline-grounded clinical recommendations."""
    if not chatbot:
        raise HTTPException(503, "Chatbot not initialized")

    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"

    try:
        result = chatbot.chat(session_id, request.message)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(500, f"Chat error: {str(e)}")


@app.get("/chat/history/{session_id}")
async def chat_history(session_id: str):
    if not chatbot:
        raise HTTPException(503, "Chatbot not initialized")
    return {"session_id": session_id, "history": chatbot.get_session_history(session_id)}


@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    if not chatbot:
        raise HTTPException(503, "Chatbot not initialized")
    chatbot.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/chat/guidelines")
async def chat_guidelines():
    if not chatbot:
        raise HTTPException(503, "Chatbot not initialized")
    return chatbot.get_guidelines_summary()


# --- Provider Switching ---


@app.get("/providers")
async def list_providers():
    """List available LLM providers with current selection."""
    return {
        "current": config.LLM_PROVIDER,
        "available": [
            {"value": "ollama-llama3", "label": "Llama 3 (Local)", "type": "local", "privacy": "full"},
            {"value": "ollama-mistral", "label": "Mistral (Local)", "type": "local", "privacy": "full"},
            {"value": "ollama-meditron", "label": "Meditron (Local/Medical)", "type": "local", "privacy": "full"},
            {"value": "gemini", "label": "Gemini (Google)", "type": "cloud", "privacy": "api"},
            {"value": "claude", "label": "Claude (Anthropic)", "type": "cloud", "privacy": "api"},
            {"value": "openai", "label": "GPT-4 (OpenAI)", "type": "cloud", "privacy": "api"},
        ],
    }


@app.post("/set_provider")
async def set_provider(request: ProviderRequest):
    """Switch the active LLM provider at runtime."""
    global agent, chatbot

    valid_providers = [p.value for p in LLMProvider]
    if request.provider not in valid_providers:
        raise HTTPException(400, f"Invalid provider. Valid: {valid_providers}")

    old_provider = config.LLM_PROVIDER
    try:
        # Update environment and config
        os.environ["LLM_PROVIDER"] = request.provider
        config.LLM_PROVIDER = request.provider

        # Reset LLM client singleton so it reinitializes
        import llm_client as lc
        lc._client_instance = None

        # Reinitialize treatment agent and chatbot
        agent = TreatmentAgent()
        agent.evaluator.load_guidelines()
        chatbot = MedicalAdvisorAgent()

        logger.info(f"Provider switched: {old_provider} -> {request.provider}")
        return {"status": "ok", "provider": request.provider, "previous": old_provider}

    except Exception as e:
        logger.error(f"Provider switch failed: {e}", exc_info=True)
        # Try to restore previous provider
        os.environ["LLM_PROVIDER"] = old_provider
        config.LLM_PROVIDER = old_provider
        import llm_client as lc
        lc._client_instance = None
        raise HTTPException(500, f"Failed to switch to {request.provider}: {str(e)}")


# --- Knowledge Graph Endpoints ---


@app.get("/knowledge_graph/stats")
async def graph_stats():
    if not agent:
        raise HTTPException(503, "System not initialized")
    return agent.evaluator.get_graph_stats()


@app.get("/knowledge_graph/disease/{name}")
async def query_disease(name: str):
    if not agent:
        raise HTTPException(503, "System not initialized")
    return agent.evaluator.query_guidelines(name)


# --- System Endpoints ---


@app.get("/metrics")
async def metrics():
    return performance_tracker.get_metrics()


@app.get("/config")
async def get_config():
    return {
        "llm_provider": config.LLM_PROVIDER,
        "model_info": model_registry.get_model_info(),
        "use_quantization": config.USE_QUANTIZATION,
        "device": config.LLM_DEVICE,
        "neo4j_uri": config.NEO4J_URI,
        "api_port": config.API_PORT,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.API_HOST, port=config.API_PORT, reload=True)
