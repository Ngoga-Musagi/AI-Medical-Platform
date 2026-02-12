"""
Model Evaluation & Error Analysis
Quantitative assessment of extraction quality with error categorization.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))
from config import config
from extraction.predictor import EntityExtractor

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def fuzzy_match(a: str, b: str) -> bool:
    """Check if two strings are a fuzzy match (case-insensitive, substring)."""
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    return a_lower == b_lower or a_lower in b_lower or b_lower in a_lower


def compute_list_f1(predicted: List[str], ground_truth: List[str]) -> Dict:
    """Compute precision, recall, F1 for two lists using fuzzy matching."""
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = 0
    matched_gt = set()
    for pred in predicted:
        for i, gt in enumerate(ground_truth):
            if i not in matched_gt and fuzzy_match(pred, gt):
                tp += 1
                matched_gt.add(i)
                break

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


class EvaluationEngine:
    """Evaluate entity extraction quality against ground truth."""

    def __init__(self):
        self.extractor = EntityExtractor()

    def load_ground_truth(self, path: str = "outputs/ground_truth.json") -> Dict:
        """Load ground truth annotations."""
        with open(path, "r") as f:
            return json.load(f)

    def load_clinical_notes(self, path: str = None) -> Dict:
        """Load clinical notes, indexed by note_id."""
        path = path or config.CLINICAL_NOTES_PATH
        with open(path, "r") as f:
            notes = json.load(f)
        return {n["note_id"]: n["text"] for n in notes}

    def run_extraction(self, notes: Dict) -> Dict:
        """Run entity extraction on all notes."""
        predictions = {}
        for note_id, text in notes.items():
            try:
                start = time.time()
                entities = self.extractor.extract_entities(text)
                latency = (time.time() - start) * 1000
                # Remove metadata for comparison
                clean = {k: v for k, v in entities.items() if k != "_metadata"}
                clean["_latency_ms"] = round(latency, 1)
                predictions[note_id] = clean
                logger.info(f"{note_id}: extracted in {latency:.0f}ms - diagnosis: {clean.get('diagnosis')}")
            except Exception as e:
                logger.error(f"{note_id}: extraction failed - {e}")
                predictions[note_id] = {
                    "age": None, "sex": "unknown", "symptoms": [],
                    "diagnosis": None, "medications": [], "_error": str(e),
                }
        return predictions

    def evaluate(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """Compute evaluation metrics."""
        age_correct = 0
        sex_correct = 0
        diagnosis_correct = 0
        total = 0
        symptom_scores = []
        medication_scores = []
        latencies = []
        errors = []

        for note_id, gt in ground_truth.items():
            pred = predictions.get(note_id)
            if not pred:
                continue

            total += 1
            if pred.get("_latency_ms"):
                latencies.append(pred["_latency_ms"])

            # Age
            age_match = pred.get("age") == gt.get("age")
            if age_match:
                age_correct += 1

            # Sex
            sex_match = pred.get("sex", "").lower() == gt.get("sex", "").lower()
            if sex_match:
                sex_correct += 1

            # Diagnosis
            diag_match = fuzzy_match(
                pred.get("diagnosis") or "", gt.get("diagnosis") or ""
            )
            if diag_match:
                diagnosis_correct += 1

            # Symptoms F1
            sym_f1 = compute_list_f1(
                pred.get("symptoms", []), gt.get("symptoms", [])
            )
            symptom_scores.append(sym_f1)

            # Medications F1
            med_f1 = compute_list_f1(
                pred.get("medications", []), gt.get("medications", [])
            )
            medication_scores.append(med_f1)

            # Track errors for error analysis
            note_errors = self._categorize_errors(note_id, pred, gt)
            if note_errors:
                errors.append({"note_id": note_id, "errors": note_errors})

        # Aggregate
        avg_sym_f1 = sum(s["f1"] for s in symptom_scores) / max(1, len(symptom_scores))
        avg_med_f1 = sum(s["f1"] for s in medication_scores) / max(1, len(medication_scores))
        avg_latency = sum(latencies) / max(1, len(latencies))

        report = {
            "total_evaluated": total,
            "metrics": {
                "age_accuracy": round(age_correct / max(1, total), 3),
                "sex_accuracy": round(sex_correct / max(1, total), 3),
                "diagnosis_accuracy": round(diagnosis_correct / max(1, total), 3),
                "symptom_f1": round(avg_sym_f1, 3),
                "medication_f1": round(avg_med_f1, 3),
                "overall_extraction_f1": round(
                    (avg_sym_f1 + avg_med_f1 + diagnosis_correct / max(1, total)) / 3, 3
                ),
            },
            "per_note_symptoms": {
                note_id: symptom_scores[i]
                for i, note_id in enumerate(ground_truth.keys())
                if i < len(symptom_scores)
            },
            "per_note_medications": {
                note_id: medication_scores[i]
                for i, note_id in enumerate(ground_truth.keys())
                if i < len(medication_scores)
            },
            "latency": {
                "avg_ms": round(avg_latency, 1),
                "min_ms": round(min(latencies), 1) if latencies else 0,
                "max_ms": round(max(latencies), 1) if latencies else 0,
            },
            "errors": errors,
            "model_provider": config.LLM_PROVIDER,
        }
        return report

    def _categorize_errors(self, note_id: str, pred: Dict, gt: Dict) -> List[Dict]:
        """Categorize extraction errors for error analysis."""
        errors = []

        # Age error
        if pred.get("age") != gt.get("age"):
            errors.append({
                "field": "age",
                "type": "WRONG_VALUE" if pred.get("age") is not None else "MISSED_ENTITY",
                "expected": gt.get("age"),
                "predicted": pred.get("age"),
            })

        # Sex error
        if (pred.get("sex") or "").lower() != (gt.get("sex") or "").lower():
            errors.append({
                "field": "sex",
                "type": "WRONG_VALUE",
                "expected": gt.get("sex"),
                "predicted": pred.get("sex"),
            })

        # Diagnosis error
        if not fuzzy_match(pred.get("diagnosis") or "", gt.get("diagnosis") or ""):
            errors.append({
                "field": "diagnosis",
                "type": "WRONG_VALUE" if pred.get("diagnosis") else "MISSED_ENTITY",
                "expected": gt.get("diagnosis"),
                "predicted": pred.get("diagnosis"),
            })

        # Missed symptoms
        for gt_sym in gt.get("symptoms", []):
            found = any(fuzzy_match(gt_sym, p) for p in pred.get("symptoms", []))
            if not found:
                errors.append({
                    "field": "symptoms",
                    "type": "MISSED_ENTITY",
                    "expected": gt_sym,
                    "predicted": None,
                })

        # Hallucinated symptoms
        for pred_sym in pred.get("symptoms", []):
            found = any(fuzzy_match(pred_sym, g) for g in gt.get("symptoms", []))
            if not found:
                errors.append({
                    "field": "symptoms",
                    "type": "HALLUCINATED_ENTITY",
                    "expected": None,
                    "predicted": pred_sym,
                })

        # Missed medications
        for gt_med in gt.get("medications", []):
            found = any(fuzzy_match(gt_med, p) for p in pred.get("medications", []))
            if not found:
                errors.append({
                    "field": "medications",
                    "type": "MISSED_ENTITY",
                    "expected": gt_med,
                    "predicted": None,
                })

        # Hallucinated medications
        for pred_med in pred.get("medications", []):
            found = any(fuzzy_match(pred_med, g) for g in gt.get("medications", []))
            if not found:
                errors.append({
                    "field": "medications",
                    "type": "HALLUCINATED_ENTITY",
                    "expected": None,
                    "predicted": pred_med,
                })

        return errors

    def generate_error_analysis_report(self, report: Dict) -> str:
        """Generate human-readable error analysis."""
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Model: {report['model_provider']}")
        lines.append(f"Notes evaluated: {report['total_evaluated']}")
        lines.append("")

        metrics = report["metrics"]
        lines.append("METRICS SUMMARY:")
        lines.append(f"  Age accuracy:       {metrics['age_accuracy']:.1%}")
        lines.append(f"  Sex accuracy:       {metrics['sex_accuracy']:.1%}")
        lines.append(f"  Diagnosis accuracy: {metrics['diagnosis_accuracy']:.1%}")
        lines.append(f"  Symptom F1:         {metrics['symptom_f1']:.3f}")
        lines.append(f"  Medication F1:      {metrics['medication_f1']:.3f}")
        lines.append(f"  Overall F1:         {metrics['overall_extraction_f1']:.3f}")
        lines.append("")

        lat = report["latency"]
        lines.append(f"LATENCY: avg={lat['avg_ms']:.0f}ms, min={lat['min_ms']:.0f}ms, max={lat['max_ms']:.0f}ms")
        lines.append("")

        # Error analysis
        lines.append("ERROR ANALYSIS:")
        error_counts = defaultdict(int)
        for entry in report["errors"]:
            for err in entry["errors"]:
                error_counts[f"{err['field']}:{err['type']}"] += 1

        if error_counts:
            for key, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {key}: {count}")
        else:
            lines.append("  No errors detected!")
        lines.append("")

        # Detailed errors (first 5)
        lines.append("DETAILED ERROR EXAMPLES (up to 5):")
        for entry in report["errors"][:5]:
            lines.append(f"\n  Note {entry['note_id']}:")
            for err in entry["errors"]:
                lines.append(f"    [{err['type']}] {err['field']}: expected={err.get('expected')}, got={err.get('predicted')}")

        lines.append("")
        lines.append("COMMON FAILURE MODES:")
        lines.append("  1. Abbreviations: 'ORS' vs 'oral rehydration solution'")
        lines.append("  2. Implicit diagnoses: 'chest exam suggest pneumonia' vs explicit 'diagnosed pneumonia'")
        lines.append("  3. Multi-word symptoms being split or merged by the LLM")
        lines.append("  4. Medication normalization: 'salbutamol nebulization' vs 'salbutamol'")
        lines.append("  5. Age extraction from non-standard formats (e.g., '45yo' vs '45 year old')")

        return "\n".join(lines)


def main():
    """Run evaluation pipeline."""
    print(f"\nRunning evaluation with {config.LLM_PROVIDER}...")
    print("This will analyze 15 clinical notes against ground truth.\n")

    engine = EvaluationEngine()
    ground_truth = engine.load_ground_truth()
    clinical_notes = engine.load_clinical_notes()

    # Filter to only ground truth notes
    eval_notes = {nid: clinical_notes[nid] for nid in ground_truth if nid in clinical_notes}
    print(f"Evaluating {len(eval_notes)} notes...\n")

    # Run extraction
    predictions = engine.run_extraction(eval_notes)

    # Evaluate
    report = engine.evaluate(predictions, ground_truth)

    # Print report
    text_report = engine.generate_error_analysis_report(report)
    print(text_report)

    # Save reports
    with open("outputs/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nJSON report saved to: outputs/evaluation_report.json")

    with open("outputs/evaluation_report.txt", "w") as f:
        f.write(text_report)
    print("Text report saved to: outputs/evaluation_report.txt")

    # Save predictions
    with open("outputs/evaluation_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    print("Predictions saved to: outputs/evaluation_predictions.json")


if __name__ == "__main__":
    main()
