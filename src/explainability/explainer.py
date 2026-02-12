"""
AI Treatment Agent + Explainability Engine
Combines Knowledge Graph + LLM Reasoning + Transparent Explanations.
Pipeline: Extract -> Semantic Match -> Compliance -> Explain
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).parent.parent))
from config import config
from llm_client import get_llm_client
from extraction.predictor import EntityExtractor
from guideline_engine.evaluator import GuidelineEvaluator
from compliance.engine import ComplianceEngine

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """Generates LLM-based explanations for clinical analysis results."""

    EXPLANATION_PROMPT = """You are a clinical decision support system providing explainable AI rationale.

Clinical Note: "{note}"

Extracted Entities:
- Age: {age}
- Sex: {sex}
- Symptoms: {symptoms}
- Diagnosis: {diagnosis}
- Medications: {medications}

Guideline Match: {disease} (confidence: {match_confidence})
- Recommended drugs: {recommended_drugs}
- Contraindicated drugs: {contraindicated_drugs}
- Required tests: {required_tests}

Compliance Score: {compliance_score} ({compliance_status})

Provide a structured clinical explanation:

1. DIAGNOSIS RATIONALE: Which specific phrases in the clinical note support the diagnosis of "{diagnosis}"?
2. MEDICATION ASSESSMENT: For each prescribed medication, explain if it is appropriate based on guidelines.
3. MISSING TESTS: What required tests should be ordered and why are they clinically important?
4. OVERALL JUDGMENT: Summarize the compliance status with actionable clinical context.

Return ONLY valid JSON:
{{
    "diagnosis_rationale": {{
        "diagnosis": "{diagnosis}",
        "supporting_evidence": ["phrase1 from note", "phrase2 from note"],
        "confidence_explanation": "explanation of why this diagnosis is supported"
    }},
    "medication_explanations": [
        {{
            "medication": "drug_name",
            "status": "recommended|contraindicated|neutral",
            "explanation": "why this medication is appropriate or not"
        }}
    ],
    "missing_tests_explanation": [
        {{
            "test": "test_name",
            "clinical_importance": "why this test matters",
            "recommendation": "specific action to take"
        }}
    ],
    "overall_rationale": "summary of clinical assessment"
}}"""

    ATTRIBUTION_PROMPT = """Given this clinical note:
"{note}"

And the extracted diagnosis: "{diagnosis}"

Rate the importance of each phrase/segment in the note for arriving at this diagnosis and extracting the clinical entities.

Return ONLY valid JSON array:
[
    {{"phrase": "exact phrase from note", "importance": <0.0-1.0>, "role": "symptom|demographic|diagnosis_mention|medication|other"}}
]"""

    def __init__(self):
        self.llm = get_llm_client()

    def explain(self, note: str, entities: Dict, guidelines: Dict, compliance: Dict) -> Dict:
        """Generate comprehensive explanation for a clinical analysis."""
        try:
            rationale = self._generate_rationale(note, entities, guidelines, compliance)
            attributions = self._generate_attributions(note, entities)

            return {
                "rationale": rationale,
                "feature_attributions": attributions,
            }
        except Exception as e:
            logger.error(f"Explainability failed: {e}", exc_info=True)
            return {
                "rationale": {"error": str(e)},
                "feature_attributions": [],
            }

    def _generate_rationale(
        self, note: str, entities: Dict, guidelines: Dict, compliance: Dict
    ) -> Dict:
        """Generate LLM-based clinical rationale."""
        prompt = self.EXPLANATION_PROMPT.format(
            note=note,
            age=entities.get("age", "unknown"),
            sex=entities.get("sex", "unknown"),
            symptoms=", ".join(entities.get("symptoms", [])) or "none extracted",
            diagnosis=entities.get("diagnosis", "unknown"),
            medications=", ".join(entities.get("medications", [])) or "none prescribed",
            disease=guidelines.get("disease", "unknown"),
            match_confidence=guidelines.get("match_confidence", 0.0),
            recommended_drugs=", ".join(guidelines.get("recommended_drugs", [])) or "none",
            contraindicated_drugs=", ".join(guidelines.get("contraindicated_drugs", [])) or "none",
            required_tests=", ".join(guidelines.get("required_tests", [])) or "none",
            compliance_score=compliance.get("overall_score", 0.0),
            compliance_status=compliance.get("status", "unknown"),
        )

        response, latency_ms = self.llm.call(prompt, max_tokens=800)
        logger.info(f"Rationale generated in {latency_ms:.0f}ms")

        # Parse JSON response
        cleaned = response.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1]

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

        return {"overall_rationale": cleaned[:500]}

    def _generate_attributions(self, note: str, entities: Dict) -> list:
        """Generate phrase-level importance attributions."""
        diagnosis = entities.get("diagnosis", "")
        if not diagnosis:
            return []

        prompt = self.ATTRIBUTION_PROMPT.format(note=note, diagnosis=diagnosis)

        try:
            response, latency_ms = self.llm.call(prompt, max_tokens=600)
            logger.info(f"Attribution generated in {latency_ms:.0f}ms")

            cleaned = response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                parts = cleaned.split("```")
                if len(parts) >= 3:
                    cleaned = parts[1]

            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(cleaned[start:end])
        except Exception as e:
            logger.warning(f"Attribution generation failed: {e}")

        return []


class TreatmentAgent:
    """
    AI Agent for clinical analysis and treatment recommendations.
    Pipeline: Extract entities -> Semantic match guidelines -> Assess compliance -> Explain.
    """

    def __init__(self):
        """Initialize all pipeline components."""
        self.extractor = EntityExtractor()
        self.evaluator = GuidelineEvaluator()
        self.compliance_engine = ComplianceEngine()
        self.explainability = ExplainabilityEngine()
        logger.info(f"TreatmentAgent initialized with {config.LLM_PROVIDER}")

    def analyze(self, clinical_note: str) -> Dict:
        """
        Complete analysis pipeline.
        Returns structured result with entities, guidelines, compliance, and explanation.
        """
        start_time = time.time()

        # Step 1: Extract entities via LLM
        try:
            entities = self.extractor.extract_entities(clinical_note)
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {
                "status": "extraction_failed",
                "error": str(e),
            }

        if not entities.get("diagnosis"):
            return {
                "status": "no_diagnosis",
                "entities": {k: v for k, v in entities.items() if k != "_metadata"},
                "extraction_metadata": entities.get("_metadata", {}),
            }

        # Step 2: Query knowledge graph with semantic matching
        guidelines = self.evaluator.query_guidelines(entities["diagnosis"])

        if not guidelines.get("guideline_found"):
            return {
                "status": "no_guideline",
                "entities": {k: v for k, v in entities.items() if k != "_metadata"},
                "diagnosis": entities["diagnosis"],
                "extraction_metadata": entities.get("_metadata", {}),
            }

        # Step 3: Assess compliance
        compliance = self.compliance_engine.assess(entities, guidelines)

        # Step 4: Generate explanation
        explanation = self.explainability.explain(
            clinical_note, entities, guidelines, compliance
        )

        total_latency = (time.time() - start_time) * 1000

        return {
            "status": compliance["status"],
            "entities": {k: v for k, v in entities.items() if k != "_metadata"},
            "guidelines": {
                "disease": guidelines.get("disease"),
                "recommended_drugs": guidelines.get("recommended_drugs", []),
                "contraindicated_drugs": guidelines.get("contraindicated_drugs", []),
                "required_tests": guidelines.get("required_tests", []),
                "match_confidence": guidelines.get("match_confidence", 0.0),
            },
            "compliance": {
                "overall_score": compliance["overall_score"],
                "medication_score": compliance["medication_score"],
                "contraindication_score": compliance["contraindication_score"],
                "test_score": compliance["test_score"],
                "breakdown": compliance["breakdown"],
                "weights": compliance["weights"],
            },
            "alerts": compliance["alerts"],
            "recommendations": compliance["recommendations"],
            "explanation": explanation,
            "extraction_metadata": entities.get("_metadata", {}),
            "total_latency_ms": round(total_latency, 1),
        }


def main():
    """Test the full pipeline."""
    config.print_config()

    agent = TreatmentAgent()
    agent.evaluator.load_guidelines()

    notes = [
        "45 year old male with fever and productive cough for five days. Diagnosed with pneumonia. Started on amoxicillin.",
        "25 year old female with dysuria and lower abdominal pain. Diagnosed urinary tract infection. Prescribed ciprofloxacin.",
    ]

    for note in notes:
        print(f"\n{'='*60}")
        print(f"Note: {note[:80]}...")
        result = agent.analyze(note)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
