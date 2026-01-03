"""
CrewAI Agent Implementation

Exports CrewAIAgent for use in benchmarks.
"""
# Always try to import - the agent class will handle missing CrewAI gracefully
try:
    from .agent import CrewAIAgent
    __all__ = ['CrewAIAgent']
except ImportError:
    # If import fails (e.g., CrewAI not installed), create a dummy class
    # that raises a helpful error when instantiated
    class CrewAIAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CrewAI is not installed. Install it with: pip install 'crewai[tools]'"
            )
    __all__ = ['CrewAIAgent']

