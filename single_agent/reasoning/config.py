CONFIG = {
    "planning": False,      # toggle planning For True, we need to determine the planning_llm. defauld is openai
    # "llm": "ollama/gpt-oss:20b",
    # "planning_llm": "ollama/gpt-oss:20b",
    # "math_judge_llm":  "gpt-4o-mini",

    # "llm": "gpt-4o-mini",
    # "planning_llm": "gpt-4o-mini",
    # "math_judge_llm":  "gpt-4o-mini",

    # "llm": "gpt-4.1",
    # "planning_llm": "gpt-4.1",
    # "math_judge_llm":  "gpt-4.1",

    # "llm": "ollama/deepseek-llm:7b",
    # "planning_llm": "ollama/deepseek-llm:7b",
    # "math_judge_llm":  "gpt-4o-mini",

    # "llm": "ollama/llama3.1:8b",
    # "planning_llm": "ollama/llama3.1:8b",
    # "math_judge_llm":  "gpt-4o-mini",

    # "llm": "ollama/qwen:7b",
    # "planning_llm": "ollama/qwen:7b",
    # "math_judge_llm":  "gpt-4o-mini",

    "llm": "ollama/phi4:14b",
    "planning_llm": "ollama/phi4:14b",
    "math_judge_llm":  "gpt-4o-mini",

    

    "results_dir": "results/planning",
    "benchmarks": ["csqa","math","gsm8k"],
    # "benchmarks": ["gsm8k", "csqa"],
    # "benchmarks": ["csqa"],
    # "benchmarks": ["math"],
    "n_gsm8k": None,         # set None for full test set, a number for a subset
    "n_csqa": None,        # set None for full test set, a number for a subset
    "n_math": 100,         # set None for full test set, a number for a subset
}

