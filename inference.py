"""Inference entry point for PSHCA.

This module exposes the checker-facing script name and the required
OpenAI-compatible client contract.
"""

import os

from baseline_inference import run_evaluation


def build_hf_client():
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["HF_TOKEN"],
    )
    model = os.environ["MODEL_NAME"]
    return client, model


def main() -> int:
    if all(os.environ.get(key) for key in ["API_BASE_URL", "HF_TOKEN", "MODEL_NAME"]):
        build_hf_client()
    return run_evaluation()


if __name__ == "__main__":
    raise SystemExit(main())