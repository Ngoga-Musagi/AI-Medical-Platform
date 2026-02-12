"""
Medical Advisor Chatbot Agent
AI-powered clinical advisor providing guideline-grounded recommendations.
Works with any configured LLM (Ollama local models recommended for medical data privacy).
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).parent.parent))
from config import config
from llm_client import get_llm_client

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class MedicalAdvisorAgent:
    """
    Clinical AI Advisor Agent for guideline-grounded medical recommendations.
    Uses the shared LLM client - works with Ollama (local), Gemini, Claude, or OpenAI.
    For medical data privacy, local models via Ollama are recommended.
    """

    SYSTEM_PROMPT = """You are a Clinical AI Advisor, an expert medical decision support assistant.
You help doctors by providing evidence-based recommendations grounded in clinical guidelines.

RULES:
1. Base ALL recommendations on the clinical guidelines provided below.
2. Clearly state when advice comes from guidelines vs. general medical knowledge.
3. Flag any contraindicated medications with a clear WARNING.
4. Always recommend required diagnostic tests from guidelines.
5. If a condition is NOT in the guidelines, say so clearly.
6. Remind clinicians that AI recommendations require clinical judgment verification.

CLINICAL GUIDELINES DATABASE:
{guidelines_context}

RESPONSE FORMAT for clinical scenarios:
**Assessment**: Identified condition(s) and reasoning
**Guideline Match**: Which specific guideline applies
**Recommended Treatment**: Drugs from guidelines
**Required Tests**: Diagnostic tests from guidelines
**Warnings**: Any contraindications or safety concerns
**Summary**: Brief actionable recommendation

For general questions, respond naturally but always offer clinical assistance."""

    CHAT_PROMPT = """{system_prompt}

CONVERSATION HISTORY:
{history}

DOCTOR'S QUERY:
{message}

Provide a helpful, guideline-grounded response following the format above.
Be concise but thorough. Always reference specific guidelines when applicable."""

    def __init__(self, guidelines_path: Optional[str] = None):
        """Initialize with guidelines knowledge base and shared LLM client."""
        self.llm = get_llm_client()
        self.guidelines = self._load_guidelines(guidelines_path or config.GUIDELINES_PATH)
        self.guidelines_context = self._format_guidelines()
        self.system_prompt = self.SYSTEM_PROMPT.format(guidelines_context=self.guidelines_context)
        self.sessions: Dict[str, List[Dict]] = {}
        logger.info(
            f"MedicalAdvisorAgent initialized with {len(self.guidelines)} guidelines, "
            f"provider={config.LLM_PROVIDER}"
        )

    def _load_guidelines(self, path: str) -> Dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load guidelines: {e}")
            return {}

    def _format_guidelines(self) -> str:
        lines = []
        for disease, info in self.guidelines.items():
            rec = ", ".join(info.get("recommended_drugs", [])) or "none specified"
            avoid = ", ".join(info.get("avoid_drugs", [])) or "none"
            tests = ", ".join(info.get("required_tests", [])) or "none required"
            lines.append(
                f"- {disease.upper()}: Recommended=[{rec}], "
                f"Contraindicated=[{avoid}], Tests=[{tests}]"
            )
        return "\n".join(lines)

    def chat(self, session_id: str, message: str) -> Dict:
        """Process a chat message and return a guideline-grounded response."""
        start_time = time.time()

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        history = self.sessions[session_id]

        history_text = ""
        if history:
            for msg in history[-10:]:
                role = msg["role"].upper()
                history_text += f"{role}: {msg['content']}\n"
        else:
            history_text = "(New conversation)"

        prompt = self.CHAT_PROMPT.format(
            system_prompt=self.system_prompt,
            history=history_text,
            message=message,
        )

        try:
            response_text, llm_latency = self.llm.call(prompt, max_tokens=1500)
        except Exception as e:
            logger.error(f"Chat LLM call failed: {e}", exc_info=True)
            return {
                "response": "I encountered an error processing your request. Please try again.",
                "session_id": session_id,
                "matched_guidelines": [],
                "guideline_details": {},
                "llm_provider": config.LLM_PROVIDER,
                "latency_ms": round((time.time() - start_time) * 1000, 1),
                "history_length": len(history),
                "error": str(e),
            }

        # Store in session
        now = datetime.now(timezone.utc).isoformat()
        history.append({"role": "doctor", "content": message, "timestamp": now})
        history.append({"role": "advisor", "content": response_text, "timestamp": now})
        self.sessions[session_id] = history

        total_latency = (time.time() - start_time) * 1000

        # Dynamic guideline detection
        matched = self._detect_referenced_guidelines(message, response_text)
        guideline_details = self._get_guideline_details(matched)

        return {
            "response": response_text,
            "session_id": session_id,
            "matched_guidelines": matched,
            "guideline_details": guideline_details,
            "llm_provider": config.LLM_PROVIDER,
            "latency_ms": round(total_latency, 1),
            "history_length": len(history),
        }

    def _detect_referenced_guidelines(self, message: str, response: str) -> List[str]:
        """Detect which guidelines were referenced in the conversation."""
        combined = (message + " " + response).lower()
        matched = []
        for disease in self.guidelines:
            disease_lower = disease.lower()
            if disease_lower in combined:
                matched.append(disease)
            else:
                # Multi-word disease matching
                words = disease_lower.split()
                if len(words) > 1 and all(w in combined for w in words):
                    matched.append(disease)
        return matched

    def _get_guideline_details(self, matched_diseases: List[str]) -> Dict:
        """Get detailed guideline info for matched diseases (dynamic references)."""
        details = {}
        for disease in matched_diseases:
            info = self.guidelines.get(disease, {})
            details[disease] = {
                "recommended_drugs": info.get("recommended_drugs", []),
                "contraindicated_drugs": info.get("avoid_drugs", []),
                "required_tests": info.get("required_tests", []),
            }
        return details

    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_guidelines_summary(self) -> Dict:
        summary = {}
        for disease, info in self.guidelines.items():
            summary[disease] = {
                "recommended_drugs": info.get("recommended_drugs", []),
                "contraindicated_drugs": info.get("avoid_drugs", []),
                "required_tests": info.get("required_tests", []),
            }
        return summary
