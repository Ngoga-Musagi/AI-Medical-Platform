"""
Batch analyze all clinical notes and save results for the dashboard.

Usage:
    python run_batch.py                         # Default: http://localhost:8000
    python run_batch.py --api http://api:8000   # Custom API URL (Docker internal)
    python run_batch.py --notes outputs/clinical_notes.json
    python run_batch.py --retries 5 --timeout 180
"""

import json
import os
import sys
import time
import argparse
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Batch analyze clinical notes")
    parser.add_argument(
        "--api",
        default=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--notes",
        default=os.getenv("CLINICAL_NOTES_PATH", "outputs/clinical_notes.json"),
        help="Path to clinical notes JSON",
    )
    parser.add_argument(
        "--output",
        default="outputs/batch_results.json",
        help="Output file path (default: outputs/batch_results.json)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries per note (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    return parser.parse_args()


def wait_for_api(api_base, max_wait=120):
    """Wait for the API to be healthy before starting batch."""
    print(f"Checking API health at {api_base}...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(f"{api_base}/health", timeout=5)
            if resp.status_code == 200:
                health = resp.json()
                status = health.get("status", "unknown")
                provider = health.get("llm_provider", "unknown")
                print(f"  API is {status} (provider: {provider})")
                if status == "healthy":
                    return True
                # Degraded is ok to proceed, just warn
                if status == "degraded":
                    print("  WARNING: API is degraded (Neo4j may still be loading)")
                    return True
        except requests.ConnectionError:
            pass
        except Exception as e:
            print(f"  Health check error: {e}")
        time.sleep(3)
    return False


def analyze_note(api_base, note, timeout=120, retries=3):
    """Analyze a single note with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{api_base}/analyze_note",
                json={"note_id": note["note_id"], "text": note["text"]},
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.json(), None
            elif resp.status_code == 503:
                # Service not ready, wait and retry
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.Timeout:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None, "Request timed out"
        except requests.ConnectionError:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return None, "Connection refused - is the API running?"
        except Exception as e:
            return None, str(e)
    return None, f"Failed after {retries} retries"


def main():
    args = parse_args()

    # Load notes
    if not os.path.exists(args.notes):
        print(f"ERROR: Clinical notes file not found: {args.notes}")
        sys.exit(1)

    with open(args.notes, "r") as f:
        notes = json.load(f)

    print(f"AI Medical Data Platform - Batch Analysis")
    print(f"==========================================")
    print(f"  Notes file:  {args.notes}")
    print(f"  Total notes: {len(notes)}")
    print(f"  API:         {args.api}")
    print(f"  Output:      {args.output}")
    print(f"  Retries:     {args.retries}")
    print(f"  Timeout:     {args.timeout}s")
    print()

    # Wait for API
    if not wait_for_api(args.api):
        print("ERROR: API is not available. Start the platform first:")
        print("  ./run.sh start")
        sys.exit(1)

    print()
    print(f"Analyzing {len(notes)} clinical notes...")
    print("Each note requires ~4 LLM calls (extraction + matching + compliance + explanation)")
    print()

    results = []
    success_count = 0
    error_count = 0
    total_time = 0

    for i, note in enumerate(notes):
        start = time.time()
        result, error = analyze_note(args.api, note, args.timeout, args.retries)
        elapsed = time.time() - start
        total_time += elapsed

        if result:
            score = result.get("compliance", {}).get("overall_score", "?")
            diag = str(result.get("entities", {}).get("diagnosis", "?"))
            alerts = len(result.get("alerts", []))
            print(f"  [{i+1:3d}/{len(notes)}] {note['note_id']:6s}: {diag:25s} | score={str(score):5s} | alerts={alerts} | {elapsed:.1f}s")
            results.append(result)
            success_count += 1
        else:
            print(f"  [{i+1:3d}/{len(notes)}] {note['note_id']:6s}: ERROR - {error} | {elapsed:.1f}s")
            results.append({
                "note_id": note["note_id"],
                "status": "error",
                "error": error,
            })
            error_count += 1

        sys.stdout.flush()

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    batch_data = {
        "total": len(notes),
        "success": success_count,
        "errors": error_count,
        "total_time_seconds": round(total_time, 1),
        "avg_time_seconds": round(total_time / len(notes), 1) if notes else 0,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(batch_data, f, indent=2)

    print()
    print(f"==========================================")
    print(f"  Done! {success_count}/{len(notes)} notes analyzed successfully")
    if error_count:
        print(f"  {error_count} errors encountered")
    print(f"  Total time: {total_time:.1f}s (avg: {total_time/len(notes):.1f}s/note)" if notes else "")
    print(f"  Results saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
