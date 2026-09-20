"""Pluggable LLM backends.

Every backend is returned as a LangChain *chat model*, so the rest of the app
just calls ``llm.stream(messages)`` and never cares which one is behind it.

    ollama        local, CPU-friendly quantised models (https://ollama.com)
    gemini        Google Gemini API (cloud; needs an API key)
    transformers  4-bit Hugging Face model on an NVIDIA GPU (Colab)
    none          no LLM - the app falls back to showing the best passages

``provider: auto`` picks the GPU model when CUDA is present, else a running
Ollama, else ``none``. It never silently picks a cloud API: Gemini has to be
chosen explicitly because it sends your retrieved notes to Google.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

PROVIDERS = ["auto", "ollama", "gemini", "transformers", "none"]


class LLMUnavailableError(RuntimeError):
    """Raised when a requested backend can't be used (message is user-facing)."""


@dataclass
class BackendStatus:
    provider: str
    available: bool
    detail: str
    model: Optional[str] = None


# --------------------------------------------------------------------- status
def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _ollama_status(config: dict) -> BackendStatus:
    cfg = config["llm"]["ollama"]
    model, base = cfg["model"], cfg["base_url"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=1.5) as resp:
            names = [m.get("name", "") for m in json.load(resp).get("models", [])]
    except Exception:
        return BackendStatus(
            "ollama", False,
            f"Ollama isn't running at {base}. Install it from https://ollama.com, "
            f"then run `ollama pull {model}`.", model,
        )
    if model not in names and f"{model}:latest" not in names:
        return BackendStatus(
            "ollama", False, f"Ollama is running but '{model}' isn't downloaded. Run `ollama pull {model}`.", model
        )
    return BackendStatus("ollama", True, f"Local model '{model}' is ready.", model)


def _gemini_status(config: dict) -> BackendStatus:
    cfg = config["llm"]["gemini"]
    model = cfg["model"]
    try:
        import langchain_google_genai  # noqa: F401
    except Exception:
        return BackendStatus("gemini", False, "Install it with `pip install langchain-google-genai`.", model)
    if not os.environ.get(cfg["api_key_env"]):
        return BackendStatus(
            "gemini", False, f"Set the {cfg['api_key_env']} environment variable (free key: aistudio.google.com).", model
        )
    return BackendStatus("gemini", True, f"Gemini '{model}' (cloud - retrieved passages are sent to Google).", model)


def _transformers_status(config: dict) -> BackendStatus:
    model = config["llm"]["transformers"]["model"]
    if not _cuda_available():
        return BackendStatus("transformers", False, "Needs an NVIDIA GPU (e.g. a Colab GPU runtime).", model)
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return BackendStatus("transformers", False, "GPU found, but `bitsandbytes` isn't installed.", model)
    return BackendStatus("transformers", True, f"4-bit '{model}' on the GPU.", model)


def detect_backends(config: dict) -> list[BackendStatus]:
    """Availability of every real backend (used by the Settings UI)."""
    return [_transformers_status(config), _ollama_status(config), _gemini_status(config)]


def resolve_provider(config: dict, provider: Optional[str] = None) -> str:
    """Turn ``auto`` into a concrete provider name (``none`` if nothing is usable)."""
    chosen = provider or config["llm"]["provider"]
    if chosen != "auto":
        return chosen
    if _transformers_status(config).available:
        return "transformers"
    if _ollama_status(config).available:
        return "ollama"
    return "none"


# -------------------------------------------------------------------- loading
def _load_ollama(config: dict):
    from langchain_ollama import ChatOllama

    llm_cfg, cfg = config["llm"], config["llm"]["ollama"]
    return ChatOllama(
        model=cfg["model"],
        base_url=cfg["base_url"],
        temperature=llm_cfg["temperature"],
        num_ctx=cfg.get("num_ctx", 4096),
        num_predict=llm_cfg["max_new_tokens"],
        keep_alive="30m",  # keep the model in RAM between questions
    )


def _load_gemini(config: dict):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm_cfg, cfg = config["llm"], config["llm"]["gemini"]
    return ChatGoogleGenerativeAI(
        model=cfg["model"],
        temperature=llm_cfg["temperature"],
        max_output_tokens=llm_cfg["max_new_tokens"],
        google_api_key=os.environ[cfg["api_key_env"]],
    )


def _load_transformers(config: dict):
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    llm_cfg, cfg = config["llm"], config["llm"]["transformers"]
    quant = None
    if cfg.get("load_in_4bit", True):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], quantization_config=quant, device_map="auto")
    model.eval()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=llm_cfg["max_new_tokens"],
        do_sample=llm_cfg["temperature"] > 0,
        temperature=max(llm_cfg["temperature"], 0.01),
        top_p=0.95,
        repetition_penalty=1.1,
        return_full_text=False,
    )
    # ChatHuggingFace applies the model's own chat template ([INST] ... for Mistral).
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))


_LOADERS = {"ollama": _load_ollama, "gemini": _load_gemini, "transformers": _load_transformers}
_STATUS = {"ollama": _ollama_status, "gemini": _gemini_status, "transformers": _transformers_status}


def load_llm(config: dict, provider: Optional[str] = None):
    """Create the chat model for ``provider`` (default: from config).

    Returns ``None`` for ``none`` / when ``auto`` finds nothing usable, so callers
    can fall back to retrieval-only mode. Raises :class:`LLMUnavailableError`
    with an actionable message if an explicitly requested backend isn't ready.
    """
    resolved = resolve_provider(config, provider)
    if resolved == "none":
        return None
    if resolved not in _LOADERS:
        raise LLMUnavailableError(f"Unknown LLM provider '{resolved}'. Choose one of: {', '.join(PROVIDERS)}.")
    status = _STATUS[resolved](config)
    if not status.available:
        raise LLMUnavailableError(status.detail)
    log.info("Loading LLM backend: %s (%s)", resolved, status.model)
    return _LOADERS[resolved](config)
