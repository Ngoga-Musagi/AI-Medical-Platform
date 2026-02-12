"""
Neo4j Knowledge Graph + Guideline Evaluator
Uses LLM-based semantic matching for diagnosis lookup.
NO rule-based string matching.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List
from neo4j import GraphDatabase

sys.path.append(str(Path(__file__).parent.parent))
from config import config
from llm_client import get_llm_client

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class GuidelineEvaluator:
    """Neo4j Knowledge Graph for clinical guidelines with LLM semantic matching."""

    MATCHING_PROMPT = """You are a medical terminology expert. Match the extracted diagnosis to the closest disease in our guideline database.

Extracted diagnosis from clinical note: "{diagnosis}"

Available diseases in guideline database:
{disease_list}

Which disease from the list is the BEST semantic match? Consider:
- Synonyms (e.g., "heart attack" = "myocardial infarction")
- Abbreviations (e.g., "UTI" = "urinary tract infection", "TB" = "tuberculosis")
- Partial matches (e.g., "type 2 diabetes" matches "diabetes mellitus")
- Medical terminology variations

If no disease is a reasonable match, return "NO_MATCH".

Return ONLY valid JSON:
{{"matched_disease": "<exact name from the list above, or NO_MATCH>", "confidence": <float 0.0 to 1.0>}}"""

    def __init__(self):
        """Initialize Neo4j connection and LLM client."""
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        self.llm = get_llm_client()
        self._disease_cache: List[str] = []
        logger.info(f"Connected to Neo4j at {config.NEO4J_URI}")

    def load_guidelines(self):
        """Load guidelines from JSON into Neo4j knowledge graph."""
        with open(config.GUIDELINES_PATH, "r") as f:
            guidelines = json.load(f)

        with self.driver.session() as session:
            # Clear existing graph
            session.run("MATCH (n) DETACH DELETE n")

            for disease, guideline in guidelines.items():
                # Create Disease node
                session.run("MERGE (d:Disease {name: $name})", name=disease)

                # Create Drug nodes and RECOMMENDED_DRUG relationships
                for drug in guideline.get("recommended_drugs", []):
                    session.run(
                        """
                        MERGE (dr:Drug {name: $drug})
                        WITH dr
                        MATCH (d:Disease {name: $disease})
                        MERGE (d)-[:RECOMMENDED_DRUG]->(dr)
                        """,
                        drug=drug,
                        disease=disease,
                    )

                # Create Drug nodes and CONTRAINDICATED_DRUG relationships
                for drug in guideline.get("avoid_drugs", []):
                    session.run(
                        """
                        MERGE (dr:Drug {name: $drug})
                        WITH dr
                        MATCH (d:Disease {name: $disease})
                        MERGE (d)-[:CONTRAINDICATED_DRUG]->(dr)
                        """,
                        drug=drug,
                        disease=disease,
                    )

                # Create Test nodes and REQUIRES_TEST relationships
                for test in guideline.get("required_tests", []):
                    session.run(
                        """
                        MERGE (t:Test {name: $test})
                        WITH t
                        MATCH (d:Disease {name: $disease})
                        MERGE (d)-[:REQUIRES_TEST]->(t)
                        """,
                        test=test,
                        disease=disease,
                    )

            # Cache disease names
            self._disease_cache = list(guidelines.keys())

        logger.info(f"Loaded {len(guidelines)} diseases into knowledge graph")

    def get_all_diseases(self) -> List[str]:
        """Get all disease names from Neo4j (cached)."""
        if self._disease_cache:
            return self._disease_cache

        with self.driver.session() as session:
            result = session.run("MATCH (d:Disease) RETURN d.name as name")
            self._disease_cache = [record["name"] for record in result]
        return self._disease_cache

    def semantic_match_disease(self, extracted_diagnosis: str) -> Dict:
        """Use LLM to semantically match diagnosis to a guideline disease."""
        diseases = self.get_all_diseases()
        if not diseases:
            return {"matched_disease": None, "confidence": 0.0}

        disease_list = "\n".join(f"- {d}" for d in diseases)
        prompt = self.MATCHING_PROMPT.format(
            diagnosis=extracted_diagnosis, disease_list=disease_list
        )

        try:
            response, latency_ms = self.llm.call(prompt, max_tokens=200)
            logger.info(f"Disease matching completed in {latency_ms:.0f}ms")

            # Parse response
            cleaned = response.strip()
            if "```" in cleaned:
                cleaned = (
                    cleaned.split("```json")[-1].split("```")[0]
                    if "```json" in cleaned
                    else cleaned.split("```")[1].split("```")[0]
                )

            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(cleaned[start:end])
                matched = parsed.get("matched_disease", "NO_MATCH")
                confidence = float(parsed.get("confidence", 0.0))

                if matched == "NO_MATCH":
                    return {"matched_disease": None, "confidence": 0.0}

                # Verify matched disease is in our list (case-insensitive)
                for d in diseases:
                    if d.lower() == matched.lower():
                        return {"matched_disease": d, "confidence": confidence}

                # LLM returned something not in the list
                return {"matched_disease": None, "confidence": 0.0}

        except Exception as e:
            logger.error(f"Semantic disease matching failed: {e}", exc_info=True)

        # Fallback: simple case-insensitive containment (last resort)
        diag_lower = extracted_diagnosis.lower()
        for d in diseases:
            if d.lower() in diag_lower or diag_lower in d.lower():
                return {"matched_disease": d, "confidence": 0.5}

        return {"matched_disease": None, "confidence": 0.0}

    def query_guidelines(self, diagnosis: str) -> Dict:
        """Query guidelines using LLM semantic matching."""
        if not diagnosis:
            return {"guideline_found": False}

        # Step 1: Semantic match to closest disease
        match_result = self.semantic_match_disease(diagnosis)
        matched_disease = match_result["matched_disease"]
        match_confidence = match_result["confidence"]

        if not matched_disease:
            logger.info(f"No guideline match for: {diagnosis}")
            return {"guideline_found": False, "query_diagnosis": diagnosis}

        logger.info(
            f"Matched '{diagnosis}' -> '{matched_disease}' (confidence: {match_confidence:.2f})"
        )

        # Step 2: Query Neo4j with exact matched name
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Disease {name: $disease})
                OPTIONAL MATCH (d)-[:RECOMMENDED_DRUG]->(rec:Drug)
                OPTIONAL MATCH (d)-[:CONTRAINDICATED_DRUG]->(contra:Drug)
                OPTIONAL MATCH (d)-[:REQUIRES_TEST]->(test:Test)
                RETURN d.name as disease,
                       collect(DISTINCT rec.name) as recommended_drugs,
                       collect(DISTINCT contra.name) as contraindicated_drugs,
                       collect(DISTINCT test.name) as required_tests
                """,
                disease=matched_disease,
            ).single()

            if result:
                return {
                    "disease": result["disease"],
                    "recommended_drugs": [d for d in result["recommended_drugs"] if d],
                    "contraindicated_drugs": [
                        d for d in result["contraindicated_drugs"] if d
                    ],
                    "required_tests": [t for t in result["required_tests"] if t],
                    "guideline_found": True,
                    "match_confidence": match_confidence,
                }

        return {"guideline_found": False, "query_diagnosis": diagnosis}

    def get_graph_stats(self) -> Dict:
        """Return statistics about the knowledge graph."""
        with self.driver.session() as session:
            diseases = session.run(
                "MATCH (d:Disease) RETURN count(d) as count"
            ).single()["count"]
            drugs = session.run("MATCH (d:Drug) RETURN count(d) as count").single()[
                "count"
            ]
            tests = session.run("MATCH (t:Test) RETURN count(t) as count").single()[
                "count"
            ]
            rels = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()["count"]
        return {
            "diseases": diseases,
            "drugs": drugs,
            "tests": tests,
            "relationships": rels,
        }

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()


def main():
    """Test guideline evaluator."""
    config.print_config()

    evaluator = GuidelineEvaluator()
    evaluator.load_guidelines()

    print("\nGraph Stats:", json.dumps(evaluator.get_graph_stats(), indent=2))

    test_cases = ["pneumonia", "UTI", "heart attack", "TB", "diabetes"]
    for diag in test_cases:
        print(f"\nQuery: {diag}")
        result = evaluator.query_guidelines(diag)
        print(json.dumps(result, indent=2))

    evaluator.close()


if __name__ == "__main__":
    main()
