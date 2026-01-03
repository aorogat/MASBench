"""
OpenAI Agents SDK Agent Package

Exports OpenAISDKAgent for use in benchmarks.
"""
try:
    from .agent import OpenAISDKAgent
    __all__ = ["OpenAISDKAgent"]
except ImportError as e:
    # If OpenAI Agents SDK is not installed, set OpenAISDKAgent to None
    # This allows the module to be imported without error
    OpenAISDKAgent = None
    __all__ = []
    import warnings
    warnings.warn(
        f"OpenAI Agents SDK is not installed: {e}. "
        "Install it with: pip install openai-agents"
    )

