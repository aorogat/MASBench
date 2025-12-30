import os
import json
import re
from glob import glob
from collections import defaultdict


RESULTS_DIR = "results/memory"
OUT_TEX = os.path.join(RESULTS_DIR, "runtime_vs_context.tex")

SYSTEMS = {
    "LangGraph": r"LangGraph_STM_LTM_CTX(\d+)_([A-Za-z_]+)",
    "OpenAI SDK": r"Openai_SDK_Groq_CTX(\d+)_([A-Za-z_]+)",
}

CATEGORY_MAP = {
    "Accurate_Retrieval": "Accurate Retrieval",
    "Test_Time_Learning": "Test-Time Learning",
    "Long_Range_Understanding": "Long-Range Understanding",
    "Conflict_Resolution": "Selective Forgetting",
}

CATEGORY_ORDER = [
    "Accurate Retrieval",
    "Test-Time Learning",
    "Long-Range Understanding",
    "Selective Forgetting",
]

COLOR_MAP = {
    "Accurate Retrieval": "blue",
    "Test-Time Learning": "orange",
    "Long-Range Understanding": "green!60!black",
    "Selective Forgetting": "red",
}


def collect_runtime_data():
    data = {
        "LangGraph": defaultdict(dict),
        "OpenAI SDK": defaultdict(dict),
    }

    for file in glob(os.path.join(RESULTS_DIR, "*.json")):
        with open(file) as f:
            j = json.load(f)

        system_name = j.get("system", "")
        split = j.get("split")
        runtime = j.get("total_runtime_sec")

        if runtime is None or split is None:
            continue

        # Determine system
        if system_name.startswith("LangGraph"):
            system = "LangGraph"
        elif system_name.startswith("Openai_SDK_Groq"):
            system = "OpenAI SDK"
        else:
            continue

        # Extract context window from system name
        m = re.search(r"CTX(\d+)", system_name)
        if not m:
            continue

        ctx = int(m.group(1))

        if split not in CATEGORY_MAP:
            continue

        category = CATEGORY_MAP[split]

        data[system][category][ctx] = runtime

    # 🔴 FAIL LOUDLY if no data
    for system in data:
        if not data[system]:
            raise RuntimeError(f"No runtime data found for {system}")

    return data


def pgfplot_lines(cat_data, color):
    xs = sorted(cat_data.keys())
    coords = " ".join(f"({x},{cat_data[x]:.1f})" for x in xs)
    return rf"\addplot+[mark=*, thick, {color}] coordinates {{{coords}}};"


def generate_latex(data):
    tex = [
        r"\begin{figure}[t]",
        r"\centering",
        r"\setlength{\abovecaptionskip}{2pt}",
        r"\setlength{\belowcaptionskip}{-6pt}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 1, horizontal sep=1.0cm},",
        r"  width=0.48\linewidth,",
        r"  height=0.55\linewidth,",
        r"  xmode=log,",
        r"  log basis x=2,",
        r"  xlabel={Context Window (tokens)},",
        r"  ylabel={Total Runtime (s)},",
        r"  grid=both,",
        r"  grid style={dashed,gray!30},",
        r"  tick label style={font=\scriptsize},",
        r"  label style={font=\scriptsize},",
        r"  legend style={font=\scriptsize, draw=none},",
        r"]",
    ]

    for system in SYSTEMS:
        tex.append(rf"\nextgroupplot[title={{{system}}}]")
        for cat in CATEGORY_ORDER:
            if cat not in data[system]:
                continue
            tex.append(pgfplot_lines(data[system][cat], COLOR_MAP[cat]))
            tex.append(rf"\addlegendentry{{{cat}}}")

    tex += [
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"\caption{Total runtime as a function of context window size for LangGraph (left) and OpenAI SDK (right) across four MemoryAgentBench task categories.}",
        r"\label{fig:runtime-vs-context}",
        r"\end{figure}",
    ]

    return "\n".join(tex)


if __name__ == "__main__":
    data = collect_runtime_data()
    latex_code = generate_latex(data)

    with open(OUT_TEX, "w") as f:
        f.write(latex_code)

    print(f"✅ LaTeX figure written to {OUT_TEX}")
