"""
Shared LLM Client - Single reusable client for all modules.
Handles Claude, OpenAI, Gemini, Ollama, and HuggingFace providers.
"""

import json
import time
import logging
import traceback
from typing import Tuple, Optional

from config import config, LLMProvider

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

try:
    from google import generativeai as genai
except ImportError:
    genai = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import ollama as ollama_lib
    from ollama import Client as OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama_lib = None
    OllamaClient = None
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMClient:
    """Shared LLM client used by all modules for LLM calls."""

    def __init__(self):
        self.provider = config.get_llm_provider()
        self.api_key = config.get_api_key()
        self._client = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._ollama_model = None
        self._ollama_client = None
        self._initialize()

    def _initialize(self):
        """Initialize the appropriate LLM client."""
        if self.provider == LLMProvider.CLAUDE:
            if not anthropic:
                raise ImportError("Claude requires: pip install anthropic")
            if not self.api_key:
                raise ValueError("Claude requires ANTHROPIC_API_KEY in .env")
            self._client = anthropic.Anthropic(api_key=self.api_key)
            logger.info(f"LLMClient initialized: Claude ({config.CLAUDE_MODEL})")

        elif self.provider == LLMProvider.OPENAI:
            if not openai:
                raise ImportError("OpenAI requires: pip install openai")
            if not self.api_key:
                raise ValueError("OpenAI requires OPENAI_API_KEY in .env")
            self._client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"LLMClient initialized: OpenAI ({config.OPENAI_MODEL})")

        elif self.provider == LLMProvider.GEMINI:
            if not genai:
                raise ImportError("Gemini requires: pip install google-generativeai")
            if not self.api_key:
                raise ValueError("Gemini requires GEMINI_API_KEY in .env")
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(config.GEMINI_MODEL)
            logger.info(f"LLMClient initialized: Gemini ({config.GEMINI_MODEL})")

        elif self.provider in [LLMProvider.OLLAMA_LLAMA3, LLMProvider.OLLAMA_MISTRAL, LLMProvider.OLLAMA_MEDITRON]:
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama requires: pip install ollama")
            model_map = {
                LLMProvider.OLLAMA_LLAMA3: "llama3",
                LLMProvider.OLLAMA_MISTRAL: "mistral",
                LLMProvider.OLLAMA_MEDITRON: "meditron",
            }
            self._ollama_model = model_map[self.provider]
            # Connect to Ollama Docker service (or external host)
            ollama_host = config.OLLAMA_HOST
            self._ollama_client = OllamaClient(host=ollama_host)
            logger.info(f"LLMClient initialized: Ollama ({self._ollama_model}) at {ollama_host}")

        else:
            self._load_huggingface_model()

    def _load_huggingface_model(self):
        """Load open-source model from Hugging Face."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("HuggingFace requires: pip install torch transformers accelerate bitsandbytes")

        model_configs = {
            LLMProvider.LLAMA_3_8B: "meta-llama/Meta-Llama-3-8B-Instruct",
            LLMProvider.LLAMA_3_70B: "meta-llama/Meta-Llama-3-70B-Instruct",
            LLMProvider.MISTRAL_7B: "mistralai/Mistral-7B-Instruct-v0.2",
            LLMProvider.MIXTRAL_8X7B: "mistralai/Mixtral-8x7B-Instruct-v0.1",
            LLMProvider.MEDITRON_7B: "epfl-llm/meditron-7b",
            LLMProvider.BIOMISTRAL_7B: "BioMistral/BioMistral-7B",
        }
        model_id = model_configs.get(self.provider)
        if not model_id:
            raise ValueError(f"Unknown provider: {self.provider}")

        logger.info(f"Loading HuggingFace model: {model_id}...")
        self._hf_tokenizer = AutoTokenizer.from_pretrained(model_id, token=config.HUGGINGFACE_TOKEN)

        model_kwargs = {}
        if config.USE_QUANTIZATION and torch.cuda.is_available():
            logger.info("Using 4-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs["device_map"] = "auto"

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, token=config.HUGGINGFACE_TOKEN, **model_kwargs
        )
        logger.info(f"HuggingFace model loaded: {model_id}")

    def call(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Call the configured LLM with a prompt.

        Returns:
            Tuple of (response_text, latency_ms)
        Raises:
            Exception on failure after retries.
        """
        max_tokens = max_tokens or config.LLM_MAX_TOKENS
        last_error = None

        for attempt in range(config.EXTRACTION_RETRY_COUNT):
            try:
                start = time.time()
                response = self._call_provider(prompt, max_tokens)
                latency_ms = (time.time() - start) * 1000
                logger.debug(f"LLM call succeeded in {latency_ms:.0f}ms (attempt {attempt + 1})")
                return response, latency_ms

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{config.EXTRACTION_RETRY_COUNT}): {e}"
                )
                if attempt < config.EXTRACTION_RETRY_COUNT - 1:
                    delay = config.EXTRACTION_RETRY_DELAY * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)

        logger.error(f"LLM call failed after {config.EXTRACTION_RETRY_COUNT} attempts:\n{traceback.format_exc()}")
        raise last_error

    def _call_provider(self, prompt: str, max_tokens: int) -> str:
        """Execute the actual LLM API call."""
        if self.provider == LLMProvider.CLAUDE:
            response = self._client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif self.provider == LLMProvider.OPENAI:
            response = self._client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        elif self.provider == LLMProvider.GEMINI:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": config.LLM_TEMPERATURE,
                    "max_output_tokens": max_tokens,
                },
            )
            return response.text

        elif self._ollama_model:
            response = self._ollama_client.generate(
                model=self._ollama_model,
                prompt=prompt,
                options={"temperature": config.LLM_TEMPERATURE},
            )
            return response["response"]

        else:
            inputs = self._hf_tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
            outputs = self._hf_model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=config.LLM_TEMPERATURE
            )
            full = self._hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return full.split(prompt)[-1]

    def get_model_info(self) -> dict:
        """Return info about the current model for logging."""
        model_names = {
            LLMProvider.CLAUDE: config.CLAUDE_MODEL,
            LLMProvider.OPENAI: config.OPENAI_MODEL,
            LLMProvider.GEMINI: config.GEMINI_MODEL,
            LLMProvider.OLLAMA_LLAMA3: "llama3",
            LLMProvider.OLLAMA_MISTRAL: "mistral",
            LLMProvider.OLLAMA_MEDITRON: "meditron",
            LLMProvider.LLAMA_3_8B: "meta-llama/Meta-Llama-3-8B-Instruct",
            LLMProvider.LLAMA_3_70B: "meta-llama/Meta-Llama-3-70B-Instruct",
            LLMProvider.MISTRAL_7B: "mistralai/Mistral-7B-Instruct-v0.2",
            LLMProvider.MIXTRAL_8X7B: "mistralai/Mixtral-8x7B-Instruct-v0.1",
            LLMProvider.MEDITRON_7B: "epfl-llm/meditron-7b",
            LLMProvider.BIOMISTRAL_7B: "BioMistral/BioMistral-7B",
        }
        return {
            "provider": self.provider.value,
            "model": model_names.get(self.provider, "unknown"),
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        }


# Module-level singleton
_client_instance = None


def get_llm_client() -> LLMClient:
    """Get or create the shared LLM client singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance
