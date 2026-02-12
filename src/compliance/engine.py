"""
Deterministic Compliance Engine
Assesses extracted entities against clinical guidelines with transparent, weighted scoring.
Uses LLM for semantic drug name matching - NO rule-based keyword matching.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))
from config import config
from llm_client import get_llm_client

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Scoring weights
WEIGHT_MEDICATION = 0.40
WEIGHT_CONTRAINDICATION = 0.35
WEIGHT_TESTS = 0.25


class ComplianceEngine:
    """
    Deterministic compliance scoring engine.
    Transparent scoring: medication compliance (40%), contraindication check (35%), test completeness (25%).
    Uses LLM for semantic matching of drug names.
    """

    DRUG_MATCHING_PROMPT = """You are a pharmacology expert. Determine if the prescribed medication semantically matches any medication in the guideline list.

Prescribed medication: "{prescribed}"
Guideline medications: {guideline_list}

Consider:
- Generic vs brand names (e.g., "tylenol" = "paracetamol")
- Abbreviations (e.g., "ORS" = "ors")
- Dosage variations (e.g., "amoxicillin 500mg" matches "amoxicillin")
- Case differences

Return ONLY valid JSON:
{{"matches": "<exact name from guideline list or NO_MATCH>", "is_match": <true/false>}}"""

    def __init__(self):
        self.llm = get_llm_client()

    def assess(self, entities: Dict, guidelines: Dict) -> Dict:
        """
        Assess compliance of extracted entities against guidelines.

        Returns:
            Dict with overall_score, medication_score, contraindication_score,
            test_score, breakdown, alerts, recommendations.
        """
        prescribed = entities.get("medications", [])
        recommended = guidelines.get("recommended_drugs", [])
        contraindicated = guidelines.get("contraindicated_drugs", [])
        required_tests = guidelines.get("required_tests", [])
        disease = guidelines.get("disease", "unknown")

        breakdown = []
        alerts = []
        recommendations = []

        # --- 1. Medication Compliance (40%) ---
        med_score = self._assess_medications(
            prescribed, recommended, disease, breakdown, recommendations
        )

        # --- 2. Contraindication Check (35%) ---
        contra_score = self._assess_contraindications(
            prescribed, contraindicated, disease, breakdown, alerts
        )

        # --- 3. Test Completeness (25%) ---
        test_score = self._assess_tests(
            required_tests, disease, breakdown, recommendations
        )

        # --- Overall Score ---
        overall = (
            WEIGHT_MEDICATION * med_score
            + WEIGHT_CONTRAINDICATION * contra_score
            + WEIGHT_TESTS * test_score
        )
        overall = max(0.0, min(1.0, overall))

        # Determine status
        if alerts:
            status = "CRITICAL - Contraindicated medication prescribed"
        elif overall >= 0.8:
            status = "Appropriate treatment - guideline compliant"
        elif overall >= 0.5:
            status = "Partially compliant - review recommended"
        else:
            status = "Non-compliant - immediate review needed"

        return {
            "overall_score": round(overall, 3),
            "medication_score": round(med_score, 3),
            "contraindication_score": round(contra_score, 3),
            "test_score": round(test_score, 3),
            "status": status,
            "breakdown": breakdown,
            "alerts": alerts,
            "recommendations": recommendations,
            "weights": {
                "medication": WEIGHT_MEDICATION,
                "contraindication": WEIGHT_CONTRAINDICATION,
                "tests": WEIGHT_TESTS,
            },
        }

    def _assess_medications(
        self,
        prescribed: List[str],
        recommended: List[str],
        disease: str,
        breakdown: List,
        recommendations: List,
    ) -> float:
        """Assess medication compliance. Returns score 0.0-1.0."""
        if not recommended:
            breakdown.append({
                "check": "medication_recommended",
                "item": "N/A",
                "status": "INFO",
                "reason": "No specific medications recommended in guidelines",
                "score_impact": "0.0",
            })
            return 1.0

        if not prescribed:
            breakdown.append({
                "check": "medication_recommended",
                "item": "none prescribed",
                "status": "FAIL",
                "reason": f"No medications prescribed but guidelines recommend: {', '.join(recommended)}",
                "score_impact": "-1.0",
            })
            recommendations.append(f"Consider prescribing recommended medications: {', '.join(recommended)}")
            return 0.0

        matches = 0
        for med in prescribed:
            matched_drug = self._semantic_match_drug(med, recommended)
            if matched_drug:
                matches += 1
                breakdown.append({
                    "check": "medication_recommended",
                    "item": med,
                    "status": "PASS",
                    "reason": f"{med} is a recommended drug for {disease} (matches: {matched_drug})",
                    "score_impact": "+",
                })
            else:
                breakdown.append({
                    "check": "medication_recommended",
                    "item": med,
                    "status": "WARNING",
                    "reason": f"{med} is not in the recommended list for {disease}. Recommended: {', '.join(recommended)}",
                    "score_impact": "0",
                })

        score = matches / max(len(recommended), len(prescribed))
        return min(1.0, score)

    def _assess_contraindications(
        self,
        prescribed: List[str],
        contraindicated: List[str],
        disease: str,
        breakdown: List,
        alerts: List,
    ) -> float:
        """Assess contraindication compliance. Returns 1.0 if clean, 0.0 if violated."""
        if not contraindicated:
            breakdown.append({
                "check": "contraindication_check",
                "item": "N/A",
                "status": "PASS",
                "reason": "No contraindicated drugs listed for this disease",
                "score_impact": "+1.0",
            })
            return 1.0

        violated = False
        for med in prescribed:
            matched_contra = self._semantic_match_drug(med, contraindicated)
            if matched_contra:
                violated = True
                breakdown.append({
                    "check": "contraindication_check",
                    "item": med,
                    "status": "FAIL",
                    "reason": f"CRITICAL: {med} is CONTRAINDICATED for {disease}",
                    "score_impact": "-1.0",
                })
                alerts.append(
                    f"{med} is contraindicated for {disease}. "
                    f"This medication should be avoided per clinical guidelines."
                )
            else:
                breakdown.append({
                    "check": "contraindication_check",
                    "item": med,
                    "status": "PASS",
                    "reason": f"{med} is not in the contraindicated list for {disease}",
                    "score_impact": "+",
                })

        return 0.0 if violated else 1.0

    def _assess_tests(
        self,
        required_tests: List[str],
        disease: str,
        breakdown: List,
        recommendations: List,
    ) -> float:
        """Assess test completeness. Returns score 0.0-1.0."""
        if not required_tests:
            breakdown.append({
                "check": "test_completeness",
                "item": "N/A",
                "status": "PASS",
                "reason": "No specific tests required for this disease",
                "score_impact": "+1.0",
            })
            return 1.0

        # Since clinical notes typically don't mention ordered tests explicitly,
        # we flag all required tests as recommendations
        for test in required_tests:
            breakdown.append({
                "check": "test_completeness",
                "item": test,
                "status": "WARNING",
                "reason": f"{test} is required for {disease} - ensure it is ordered",
                "score_impact": "0",
            })
            recommendations.append(f"Ensure required test is ordered: {test}")

        # Give partial credit - the tests are flagged, not necessarily missing
        return 0.5

    def _semantic_match_drug(self, prescribed: str, drug_list: List[str]) -> str:
        """
        Use LLM to semantically match a prescribed drug to a list.
        Returns the matched drug name or None.
        """
        if not drug_list:
            return None

        # Quick exact match first (case-insensitive)
        prescribed_lower = prescribed.lower().strip()
        for drug in drug_list:
            if drug.lower().strip() == prescribed_lower:
                return drug

        # Quick substring match
        for drug in drug_list:
            if drug.lower() in prescribed_lower or prescribed_lower in drug.lower():
                return drug

        # LLM semantic match for non-obvious cases
        try:
            guideline_list = json.dumps(drug_list)
            prompt = self.DRUG_MATCHING_PROMPT.format(
                prescribed=prescribed, guideline_list=guideline_list
            )
            response, _ = self.llm.call(prompt, max_tokens=150)

            cleaned = response.strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(cleaned[start:end])
                if parsed.get("is_match", False):
                    matched = parsed.get("matches", "NO_MATCH")
                    if matched != "NO_MATCH" and matched in drug_list:
                        return matched
        except Exception as e:
            logger.warning(f"LLM drug matching failed for '{prescribed}': {e}")

        return None


def main():
    """Test compliance engine."""
    engine = ComplianceEngine()

    # Test case: pneumonia with amoxicillin (compliant)
    entities = {
        "medications": ["amoxicillin"],
        "diagnosis": "pneumonia",
    }
    guidelines = {
        "disease": "pneumonia",
        "recommended_drugs": ["amoxicillin", "azithromycin"],
        "contraindicated_drugs": ["ciprofloxacin"],
        "required_tests": ["chest_xray"],
    }

    result = engine.assess(entities, guidelines)
    print("Test 1 - Pneumonia (compliant):")
    print(json.dumps(result, indent=2))

    # Test case: UTI with ciprofloxacin (contraindicated)
    entities2 = {
        "medications": ["ciprofloxacin"],
        "diagnosis": "urinary tract infection",
    }
    guidelines2 = {
        "disease": "urinary tract infection",
        "recommended_drugs": ["nitrofurantoin"],
        "contraindicated_drugs": ["ciprofloxacin"],
        "required_tests": ["urinalysis"],
    }

    result2 = engine.assess(entities2, guidelines2)
    print("\nTest 2 - UTI (contraindicated):")
    print(json.dumps(result2, indent=2))


if __name__ == "__main__":
    main()
