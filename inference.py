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

import os

# Only load .env locally. The hackathon validator injects these securely on remote servers.
if "API_BASE_URL" not in os.environ:
    from dotenv import load_dotenv
    load_dotenv()

from baseline_inference import run_evaluation  # type: ignore


def build_hf_client():
    """
    Construct and return an (OpenAI client, model_name) tuple using the
    mandatory hackathon environment variables.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", ""),
    )
    model = os.environ.get("MODEL_NAME", "default")
    return client, model


def main() -> int:
    """
    Entry point called by the hackathon checker.

    - Validates env vars are present (does not crash if missing — falls back
      to demo mode inside run_evaluation).
    - Delegates all evaluation logic to baseline_inference.run_evaluation().
    - Returns exit code 0 (pass) or 1 (fail).
    """
    if "API_BASE_URL" in os.environ and ("API_KEY" in os.environ or "HF_TOKEN" in os.environ):
        # Warm up the client to surface credential errors early
        build_hf_client()

    return run_evaluation()


if __name__ == "__main__":
    raise SystemExit(main())