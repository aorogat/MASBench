"""
Generate a LaTeX (PGFPlots/TikZ) figure from framework overhead JSON results.

The .tex file is saved in the SAME directory as the JSON input.

Usage:
    python single_agent/framework_overhead/latex_plot_framework_overhead.py \
        --input results/framework_overhead/framework_overhead_50_TRIALS.json
"""

import json
import argparse
import os
import math


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------
def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = {}
    for r in raw:
        name = r["name"]
        r["_full_name"] = name
        data[name] = r

    return data


# ------------------------------------------------------------
# LaTeX generation (SORTED BY p50 LATENCY)
# ------------------------------------------------------------
def generate_latex(data):
    # --------------------------------------------------------
    # Sort frameworks by FIRST FIGURE: p50 latency (small → big)
    # --------------------------------------------------------
    frameworks = sorted(
        data.keys(),
        key=lambda f: data[f]["p50_latency"]
    )

    xcoords = ",".join(frameworks)

    # ---- Metrics ----
    p50 = {f: data[f]["p50_latency"] / 1000 for f in frameworks}
    p95 = {f: data[f]["p95_latency"] / 1000 for f in frameworks}
    throughput = {f: data[f]["throughput_req_per_sec"] for f in frameworks}
    out_mean = {f: data[f]["output_chars_mean"] for f in frameworks}
    out_min = {f: data[f]["output_chars_min"] for f in frameworks}
    out_max = {f: data[f]["output_chars_max"] for f in frameworks}
    out_total = {f: data[f]["output_chars_total"] for f in frameworks}

    # ---- Dynamic ymax (+15%) ----
    ymax_latency = 1.15 * max(max(p50.values()), max(p95.values()))
    ymax_throughput = 1.15 * max(throughput.values())
    ymax_mean_out = 1.15 * max(out_mean.values())
    ymax_out_summary = 1.15 * max(
        max(out_min.values()),
        max(out_max.values()),
        max(out_total.values())
    )

    coords = lambda d: " ".join(f"({k},{v:.3f})" for k, v in d.items())

    caption_defs = ", ".join(rf"\textbf{{{k}}}" for k in frameworks)

    axis_common = r"""tick label style={font=\scriptsize},
    ylabel style={font=\scriptsize},
    xlabel style={font=\scriptsize},
    ybar,
    bar width=5pt,
    enlarge x limits=0.15,
    ymajorgrids=true,
    grid style={dashed,gray!30},
    nodes near coords,
    nodes near coords style={
        font=\scriptsize,
        yshift=7pt,
        xshift=5pt,
        anchor=south,
        rotate=90
    },
    every axis plot/.append style={fill opacity=0.9}"""

    latex = rf"""
\begin{{figure}}[t]
\centering

% ---------------- Latency ----------------
\begin{{subfigure}}{{0.49\linewidth}}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=1.25\linewidth,
    height=4cm,
    ymin=0,
    ymax={ymax_latency:.3f},
    symbolic x coords={{{xcoords}}},
    xtick=data,
    xticklabel style={{rotate=30, anchor=east, font=\scriptsize}},
    ylabel={{Latency (s)}},
    ylabel style={{at={{(axis description cs:1.08,0.2)}}, anchor=west}},
    legend style={{at={{(0.02,0.98)}}, anchor=north west, font=\scriptsize}},
    {axis_common}
]
\addplot+[fill=blue!60] coordinates {{{coords(p50)}}};
\addplot+[fill=green!60] coordinates {{{coords(p95)}}};
\legend{{p50, p95}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Latency}}
\end{{subfigure}}
\hfill

% ---------------- Throughput ----------------
\begin{{subfigure}}{{0.49\linewidth}}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=1.15\linewidth,
    height=4cm,
    ymin=0,
    ymax={ymax_throughput:.3f},
    symbolic x coords={{{xcoords}}},
    xtick=data,
    xticklabel style={{rotate=30, anchor=east, font=\scriptsize}},
    ylabel={{Throughput (req/s)}},
    ylabel style={{at={{(axis description cs:1.08,-0.05)}}, anchor=west}},
    {axis_common}
]
\addplot+[fill=red!60] coordinates {{{coords(throughput)}}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{Throughput}}
\end{{subfigure}}

\vspace{{0.6em}}

% ---------------- Mean Output ----------------
\begin{{subfigure}}{{0.49\linewidth}}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=1.05\linewidth,
    height=4cm,
    ymode=log,
    log basis y=10,
    ymin=1,
    ymax={ymax_mean_out:.3f},
    symbolic x coords={{{xcoords}}},
    xtick=data,
    xticklabel style={{rotate=30, anchor=east, font=\scriptsize}},
    ylabel={{Output size (chars)}},
    ylabel style={{at={{(axis description cs:1.08,0.2)}}, anchor=west}},
    {axis_common}
]
\addplot+[fill=orange!60] coordinates {{{coords(out_mean)}}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{Mean output}}
\end{{subfigure}}
\hfill

% ---------------- Output Summary ----------------
\begin{{subfigure}}{{0.49\linewidth}}
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
    width=1.25\linewidth,
    height=4cm,
    ymode=log,
    log basis y=10,
    ymin=1,
    ymax={ymax_out_summary:.3f},
    symbolic x coords={{{xcoords}}},
    xtick=data,
    xticklabel style={{rotate=30, anchor=east, font=\scriptsize}},
    ylabel={{Output size (chars)}},
    ylabel style={{at={{(axis description cs:1.08,0.2)}}, anchor=west}},
    legend style={{at={{(0.02,0.98)}}, anchor=north west, font=\scriptsize}},
    {axis_common}
]
\addplot+[fill=blue!60] coordinates {{{coords(out_min)}}};
\addplot+[fill=red!60] coordinates {{{coords(out_max)}}};
\addplot+[fill=orange!60] coordinates {{{coords(out_total)}}};
\legend{{Min, Max, Total}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Output summary}}
\end{{subfigure}}

\caption{{Framework overhead results for 50 trials of the trivial task (``What is 2+2?''). 
Frameworks are ordered by increasing p50 latency.
{caption_defs}.}}
\label{{fig:framework-overhead}}
\end{{figure}}
"""
    return latex.strip()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON results file")
    args = parser.parse_args()

    json_path = args.input
    out_dir = os.path.dirname(json_path)
    out_tex = os.path.join(out_dir, "framework_overhead.tex")

    data = load_results(json_path)
    latex_code = generate_latex(data)

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f"📄 LaTeX figure saved to: {out_tex}")

    print("📊 Framework order (sorted by p50 latency):")
    for f in sorted(data.keys(), key=lambda x: data[x]["p50_latency"]):
        print(f"  {f}")
