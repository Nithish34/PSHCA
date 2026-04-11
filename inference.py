"""Inference entry point for PSHCA.

This module exposes the checker-facing script name and the required
OpenAI-compatible client contract, as specified by the hackathon.

Required environment variables (hackathon validator injects these):
    API_BASE_URL  – Base URL for the OpenAI-compatible LLM proxy (LiteLLM)
    API_KEY       – Shared API key provided by the hackathon proxy
    MODEL_NAME    – Model identifier (e.g. "gpt-4o-mini")

Local development fallback:
    HF_TOKEN      – Hugging Face token (used when API_KEY is not set)
"""
from dotenv import load_dotenv
load_dotenv()

import os

from baseline_inference import run_evaluation  # type: ignore


def _get_api_key() -> str:
    """Return API key: prefer hackathon-injected API_KEY, fall back to HF_TOKEN."""
    return os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")


def build_hf_client():
    """
    Construct and return an (OpenAI client, model_name) tuple using the
    mandatory hackathon environment variables.

    The hackathon proxy injects API_BASE_URL + API_KEY + MODEL_NAME.
    For local dev, HF_TOKEN is accepted in place of API_KEY.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=_get_api_key(),
    )
    model = os.environ["MODEL_NAME"]
    return client, model


def main() -> int:
    """
    Entry point called by the hackathon checker.

    - Validates env vars are present (does not crash if missing — falls back
      to demo mode inside run_evaluation).
    - Delegates all evaluation logic to baseline_inference.run_evaluation().
    - Returns exit code 0 (pass) or 1 (fail).
    """
    # Accept API_KEY (hackathon-injected) or HF_TOKEN (local dev)
    has_key = bool(os.environ.get("API_KEY") or os.environ.get("HF_TOKEN"))
    if os.environ.get("API_BASE_URL") and has_key and os.environ.get("MODEL_NAME"):
        # Warm up the client to surface credential errors early
        build_hf_client()

    return run_evaluation()


if __name__ == "__main__":
    raise SystemExit(main())