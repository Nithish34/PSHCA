"""Inference entry point for PSHCA.

This module exposes the checker-facing script name and the required
OpenAI-compatible client contract, as specified by the hackathon.

Required environment variables:
    API_BASE_URL  – Base URL for the OpenAI-compatible LLM endpoint
    HF_TOKEN      – Hugging Face / API key
    MODEL_NAME    – Model identifier (e.g. "gpt-4o-mini")
"""
from dotenv import load_dotenv
load_dotenv()

import os

from baseline_inference import run_evaluation  # type: ignore


def build_hf_client():
    """
    Construct and return an (OpenAI client, model_name) tuple using the
    mandatory hackathon environment variables.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["HF_TOKEN"],
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
    if all(os.environ.get(key) for key in ["API_BASE_URL", "HF_TOKEN", "MODEL_NAME"]):
        # Warm up the client to surface credential errors early
        build_hf_client()

    return run_evaluation()


if __name__ == "__main__":
    raise SystemExit(main())