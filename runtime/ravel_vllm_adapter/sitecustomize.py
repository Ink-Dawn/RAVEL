"""Opt-in startup hook for the repository-owned RAVEL vLLM adapter."""
import os


if os.environ.get("RAVEL_VLLM_ADAPTER", "0") == "1":
    from ravel_engine_adapter.vllm_patch import install_from_environment

    install_from_environment()
