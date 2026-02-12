"""
Multi-LLM Entity Extractor
Uses shared LLMClient for all provider calls.
Pure semantic understanding - NO rule-based extraction.
"""

import json
import logging
import re
import sys
from typing import Dict, Any
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from llm_client import get_llm_client

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Entity extractor using configured LLM provider.
    Pure semantic understanding - NO rule-based extraction.
    """

    EXTRACTION_PROMPT = """You are a medical AI assistant. Extract structured clinical entities from the following clinical note using SEMANTIC UNDERSTANDING (not keyword matching).

Extract these fields:
1. Age (integer in years, or null if not mentioned)
2. Sex (male/female/unknown)
3. Symptoms (list of symptoms described)
4. Diagnosis (the primary clinical diagnosis)
5. Medications (list of medications prescribed or administered)

Here are examples:

Example 1:
Note: "30 year old female presenting with headache, neck stiffness and fever. Impression meningitis. Given ceftriaxone."
Output: {{"age": 30, "sex": "female", "symptoms": ["headache", "neck stiffness", "fever"], "diagnosis": "meningitis", "medications": ["ceftriaxone"]}}

Example 2:
Note: "60 year old male with chest pain radiating to left arm. Diagnosed myocardial infarction. Started aspirin and atorvastatin."
Output: {{"age": 60, "sex": "male", "symptoms": ["chest pain radiating to left arm"], "diagnosis": "myocardial infarction", "medications": ["aspirin", "atorvastatin"]}}

Example 3:
Note: "5 year old male with fever, vomiting and diarrhea. Diagnosed gastroenteritis. Started ORS."
Output: {{"age": 5, "sex": "male", "symptoms": ["fever", "vomiting", "diarrhea"], "diagnosis": "gastroenteritis", "medications": ["ors"]}}

Now extract from this clinical note:
"{clinical_note}"

Return ONLY valid JSON with no extra text:
{{"age": <int or null>, "sex": "<male/female/unknown>", "symptoms": ["symptom1"], "diagnosis": "<diagnosis>", "medications": ["med1"]}}"""

    def __init__(self):
        """Initialize using shared LLM client."""
        self.llm = get_llm_client()
        logger.info(f"EntityExtractor initialized with provider: {self.llm.provider.value}")

    def extract_entities(self, clinical_note: str) -> Dict[str, Any]:
        """Extract entities using configured LLM. Raises on fatal errors."""
        prompt = self.EXTRACTION_PROMPT.format(clinical_note=clinical_note)

        try:
            response, latency_ms = self.llm.call(prompt)
            logger.info(f"Entity extraction completed in {latency_ms:.0f}ms")

            entities = self._parse_response(response)
            validated = self._validate_entities(entities)
            validated["_metadata"] = {
                "model": self.llm.get_model_info(),
                "latency_ms": round(latency_ms, 1),
            }
            return validated

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            raise

    def _parse_response(self, response: str) -> Dict:
        """Parse JSON from LLM response with robust handling."""
        cleaned = response.strip()

        # Strip markdown code fences
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1]

        # Extract JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try fixing common issues: trailing commas, single quotes
            fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
            fixed = fixed.replace("'", '"')
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response[:300]}")
                return self._empty_entities()

    def _validate_entities(self, entities: Dict) -> Dict:
        """Clean and normalize extracted entities."""
        validated = {
            "age": entities.get("age"),
            "sex": entities.get("sex"),
            "symptoms": entities.get("symptoms", []),
            "diagnosis": entities.get("diagnosis"),
            "medications": entities.get("medications", []),
        }

        # Normalize age
        if validated["age"] is not None:
            try:
                validated["age"] = int(validated["age"])
            except (ValueError, TypeError):
                validated["age"] = None

        # Normalize sex
        if validated["sex"]:
            sex_lower = str(validated["sex"]).lower().strip()
            if "female" in sex_lower:
                validated["sex"] = "female"
            elif "male" in sex_lower:
                validated["sex"] = "male"
            else:
                validated["sex"] = "unknown"
        else:
            validated["sex"] = "unknown"

        # Normalize diagnosis
        if validated["diagnosis"]:
            validated["diagnosis"] = str(validated["diagnosis"]).strip().lower()

        # Normalize lists
        validated["symptoms"] = [s.strip().lower() for s in validated["symptoms"] if s and str(s).strip()]
        validated["medications"] = [m.strip().lower() for m in validated["medications"] if m and str(m).strip()]

        return validated

    def _empty_entities(self) -> Dict:
        """Return empty structure."""
        return {
            "age": None,
            "sex": "unknown",
            "symptoms": [],
            "diagnosis": None,
            "medications": [],
        }


def main():
    """Test extraction with configured provider."""
    print("\nCurrent Configuration:")
    config.print_config()

    if not config.validate():
        print("\n Configuration invalid. Check .env file.")
        return

    extractor = EntityExtractor()

    notes = [
        "45 year old male with fever and productive cough for five days. Diagnosed with pneumonia. Started on amoxicillin.",
        "25 year old female with dysuria and lower abdominal pain. Diagnosed urinary tract infection. Prescribed ciprofloxacin.",
        "50 year old male with cough, weight loss and night sweats. Suspected tuberculosis. Started rifampicin and isoniazid.",
    ]

    for note in notes:
        print(f"\n{'='*60}")
        print(f"Note: {note[:80]}...")
        print(f"{'='*60}")
        entities = extractor.extract_entities(note)
        # Remove metadata for display
        display = {k: v for k, v in entities.items() if k != "_metadata"}
        print(json.dumps(display, indent=2))


if __name__ == "__main__":
    main()
