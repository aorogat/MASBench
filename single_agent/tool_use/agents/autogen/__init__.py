"""
AutoGen Agent Package

Exports AutoGenAgent for use in benchmarks.
"""
try:
    from .agent import AutoGenAgent
    __all__ = ["AutoGenAgent"]
except ImportError as e:
    # If AutoGen is not installed, set AutoGenAgent to None
    # This allows the module to be imported without error
    AutoGenAgent = None
    __all__ = []
    import warnings
    warnings.warn(
        f"AutoGen AgentChat is not installed: {e}. "
        "Install it with: pip install autogen-agentchat autogen-core"
    )

