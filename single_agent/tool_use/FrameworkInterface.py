class FrameworkInterface:
    """
    A unified interface so any framework (LangGraph, CrewAI, Concordia)
    can be evaluated using the StableToolBench evaluation pipeline.
    """

    def setup_tools(self, tools: dict):
        """
        tools: dict {tool_name: tool_definition_json}
        Called once before evaluation.
        """
        raise NotImplementedError

    def reset(self):
        """Reset conversation / memory / internal state."""
        raise NotImplementedError

    def answer(self, query: str) -> str:
        """
        Answer the query using your framework.
        Must return only the final answer (string).
        """
        raise NotImplementedError
