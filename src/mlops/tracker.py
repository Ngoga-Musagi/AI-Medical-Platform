"""
MLOps Tracking - Prediction logging, model versioning, performance tracking.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import config

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Logs every prediction for audit and analysis."""

    def __init__(self):
        self.log_dir = config.PREDICTION_LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)

    def log_prediction(
        self,
        note_id: str,
        input_text: str,
        result: Dict,
        latency_ms: float,
    ):
        """Write a structured log entry as JSON line."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note_id": note_id,
            "input_length": len(input_text),
            "model_provider": config.LLM_PROVIDER,
            "status": result.get("status", "unknown"),
            "diagnosis": result.get("entities", {}).get("diagnosis"),
            "compliance_score": result.get("compliance", {}).get("overall_score"),
            "alerts_count": len(result.get("alerts", [])),
            "recommendations_count": len(result.get("recommendations", [])),
            "total_latency_ms": round(latency_ms, 1),
        }

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"predictions_{date_str}.jsonl")

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")


class ModelRegistry:
    """Track model versions and configurations."""

    MODEL_VERSIONS = {
        "claude": {"name": "claude-sonnet-4-20250514", "version": "4.0", "type": "api"},
        "openai": {"name": "gpt-4-turbo-preview", "version": "4-turbo", "type": "api"},
        "gemini": {"name": "gemini-1.5-pro", "version": "1.5", "type": "api"},
        "llama-3-8b": {"name": "meta-llama/Meta-Llama-3-8B-Instruct", "version": "3-8b", "type": "open-source"},
        "llama-3-70b": {"name": "meta-llama/Meta-Llama-3-70B-Instruct", "version": "3-70b", "type": "open-source"},
        "mistral-7b": {"name": "mistralai/Mistral-7B-Instruct-v0.2", "version": "7b-v0.2", "type": "open-source"},
        "meditron-7b": {"name": "epfl-llm/meditron-7b", "version": "7b", "type": "medical"},
    }

    def get_model_info(self) -> Dict:
        """Return current model info."""
        provider = config.LLM_PROVIDER
        info = self.MODEL_VERSIONS.get(provider, {
            "name": "unknown",
            "version": "unknown",
            "type": "unknown",
        })
        info["provider"] = provider
        return info


class PerformanceTracker:
    """Track inference latency and throughput."""

    def __init__(self, window_size: int = 1000):
        self.latencies: List[float] = []
        self.error_count: int = 0
        self.success_count: int = 0
        self.window_size = window_size
        self._start_time = time.time()

    def record(self, latency_ms: float, success: bool):
        """Record a request outcome."""
        self.latencies.append(latency_ms)
        if len(self.latencies) > self.window_size:
            self.latencies = self.latencies[-self.window_size:]
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        total = self.success_count + self.error_count
        uptime_s = time.time() - self._start_time

        sorted_lat = sorted(self.latencies) if self.latencies else [0]
        return {
            "total_requests": total,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": round(self.success_count / max(1, total), 3),
            "avg_latency_ms": round(sum(self.latencies) / max(1, len(self.latencies)), 1),
            "p50_latency_ms": round(sorted_lat[len(sorted_lat) // 2], 1),
            "p95_latency_ms": round(sorted_lat[int(0.95 * len(sorted_lat))], 1),
            "p99_latency_ms": round(sorted_lat[min(int(0.99 * len(sorted_lat)), len(sorted_lat) - 1)], 1),
            "uptime_seconds": round(uptime_s, 0),
            "model_provider": config.LLM_PROVIDER,
        }
