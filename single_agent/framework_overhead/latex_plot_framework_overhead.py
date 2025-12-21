"""
Generate a LaTeX (PGFPlots/TikZ) figure from framework overhead JSON results.

The .tex file is saved in the SAME directory as the JSON input.

Usage:
    python single_agent/framework_overhead/latex_plot_framework_overhead.py \
        --input results/framework_overhead/framework_overhead_50_TRIALS.json \
        [--order DL,LG,CR,OA,CO]
"""

import json
import argparse
import os


# ------------------------------------------------------------
# Preferred display order (soft hint for stable visual ordering)
# ------------------------------------------------------------
PREFERRED_ORDER = ["DL", "LG", "CR", "OA", "CO"]


# ------------------------------------------------------------
# Auto-generate abbreviation from framework name
# ------------------------------------------------------------
def generate_abbreviation(name):
    """
    Generate a stable abbreviation from a framework name.

    Priority:
    1) Capital letters (OpenAgents -> OA, CrewAI -> CR)
    2) Initials of words (Multi Agent Flow -> MAF)
    3) First 2 letters fallback
    """
    # Normalize
    clean = name.replace("-", " ").replace("_", " ").strip()
    words = clean.split()

    # Case 1: CamelCase / caps inside word
    caps = [c for c in name if c.isupper()]
    if len(caps) >= 2:
        return "".join(caps[:2])

    # Case 2: Multi-word name
    if len(words) > 1:
        return "".join(w[0].upper() for w in words if w)

    # Case 3: Fallback
    return name[:2].upper()


# ------------------------------------------------------------
# Load & normalize results with automatic abbreviations
# ------------------------------------------------------------
def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = {}
    used_abbrevs = set()
    
    for r in raw:
        full_name = r["name"]
        abbrev = generate_abbreviation(full_name)
        
        # Enforce uniqueness
        base = abbrev
        i = 2
        while abbrev in used_abbrevs:
            abbrev = f"{base}{i}"
            i += 1
        
        used_abbrevs.add(abbrev)
        
        # Attach metadata
        r["_abbr"] = abbrev
        r["_full_name"] = full_name
        
        data[abbrev] = r
    
    return data


# ------------------------------------------------------------
# LaTeX generation
# ------------------------------------------------------------
def generate_latex(data, custom_order=None):
    # Dynamically detect frameworks from data
    if custom_order:
        # Use custom order if provided
        frameworks = [f for f in custom_order if f in data]
    else:
        # Use preferred order as soft hint, filtering to only those in data
        frameworks = [f for f in PREFERRED_ORDER if f in data]
        # Add any new frameworks not in preferred order
        frameworks += [f for f in data if f not in frameworks]

    # Generate symbolic x coords for PGFPlots
    xcoords = ",".join(frameworks)

    # ---- Metrics ----
    p50 = {f: data[f]["p50_latency"] / 1000 for f in frameworks}
    p95 = {f: data[f]["p95_latency"] / 1000 for f in frameworks}
    throughput = {f: data[f]["throughput_req_per_sec"] for f in frameworks}
    out_mean = {f: data[f]["output_chars_mean"] for f in frameworks}
    out_min = {f: data[f]["output_chars_min"] for f in frameworks}
    out_max = {f: data[f]["output_chars_max"] for f in frameworks}
    out_total = {f: data[f]["output_chars_total"] for f in frameworks}

    coords = lambda d: " ".join(f"({k},{v:.3f})" for k, v in d.items())

    # Generate caption definitions dynamically from actual data
    caption_defs = ", ".join(
        rf"\textbf{{{k}}} = {data[k]['_full_name']}"
        for k in frameworks
    )

    # Common axis styling
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
    width=1.2\linewidth,
    height=4cm,
    ymin=0,
    symbolic x coords={{{xcoords}}},
    xtick=data,
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
    width=1.1\linewidth,
    height=4cm,
    ymin=0,
    symbolic x coords={{{xcoords}}},
    xtick=data,
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
    width=1.0\linewidth,
    height=4cm,
    ymode=log,
    log basis y=10,
    symbolic x coords={{{xcoords}}},
    xtick=data,
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
    width=1.2\linewidth,
    height=4cm,
    ymode=log,
    log basis y=10,
    symbolic x coords={{{xcoords}}},
    xtick=data,
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
    parser.add_argument("--order", help="Custom framework order (comma-separated, e.g., DL,LG,CR)")
    args = parser.parse_args()

    json_path = args.input
    out_dir = os.path.dirname(json_path)
    out_tex = os.path.join(out_dir, "framework_overhead.tex")

    # Parse custom order if provided
    custom_order = None
    if args.order:
        custom_order = [f.strip() for f in args.order.split(",")]

    data = load_results(json_path)
    
    # Validation: ensure consistency
    if len(data) != len(set(data.keys())):
        raise ValueError("Duplicate abbreviations detected!")
    
    latex_code = generate_latex(data, custom_order)

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f"📄 LaTeX figure saved to: {out_tex}")
    
    # Show detected frameworks and abbreviations
    frameworks = custom_order if custom_order else [f for f in PREFERRED_ORDER if f in data]
    frameworks += [f for f in data if f not in frameworks]
    
    print(f"📊 Frameworks detected: {len(frameworks)}")
    for abbrev in frameworks:
        full_name = data[abbrev]['_full_name']
        print(f"   {abbrev:4s} = {full_name}")