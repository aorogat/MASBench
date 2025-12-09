from StableToolBench.inference.model.model_adapter import BaseModelAdapter

class FrameworkAdapter(BaseModelAdapter):
    """
    Adapter that wraps your FrameworkInterface and exposes
    the ToolBench-compatible model API: reset(), set_tools(), chat().
    """

    def __init__(self, framework_instance):
        super().__init__()
        self.framework = framework_instance

    def reset(self):
        self.framework.reset()

    def set_tools(self, tools):
        # tools is a dict: {tool_name: tool_json}
        self.framework.setup_tools(tools)

    def chat(self, query, history=None):
        """
        Expected return format:
        {
            "response": <final_string>,
            "tool_calls": []
        }
        """
        ans = self.framework.answer(query)
        return {"response": ans, "tool_calls": []}
