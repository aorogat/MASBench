import os
import re
from datasets import load_dataset
from huggingface_hub import snapshot_download
from collections import Counter

# ============================================================
# 1. DOWNLOAD + LOAD VALIDATION SPLIT
# ============================================================

print("📥 Downloading GAIA ...")
data_dir = snapshot_download("gaia-benchmark/GAIA", repo_type="dataset")

subset = "2023_all"
split = "validation"

ds = load_dataset(path=data_dir, name=subset, split=split)
print(f"✔ Loaded {len(ds)} validation rows\n")


# ============================================================
# 2. TOOL CATEGORY NORMALIZATION
# ============================================================

def normalize_tool(t):
    t = t.lower()

    if any(k in t for k in ["browser", "youtube", "google maps", "internet", "street view", "web"]):
        return "Web Browsing"
    if "search" in t:
        return "Search Engine"
    if any(k in t for k in ["excel", "csv", "xlsx", "pdf", "xml", "json", "file"]):
        return "File Reading"
    if any(k in t for k in ["python", "code", "calculator", "compiler", "algebra", "count"]):
        return "Coding"
    if any(k in t for k in ["image", "ocr", "vision", "audio", "video", "speech"]):
        return "Multi-Modality"

    return "Other"


# ============================================================
# 3. STATISTICS
# ============================================================

total_q = len(ds)
level_dist = Counter()
attachment_types = Counter()
tool_categories = Counter()

total_attachments = 0

for ex in ds:
    lvl = ex.get("Level", None)
    if lvl is not None:
        level_dist[int(lvl)] += 1

    fp = ex.get("file_path")
    if fp:
        total_attachments += 1
        ext = os.path.splitext(fp)[1].lower()
        attachment_types[ext] += 1

    tools_raw = ex.get("Annotator Metadata", {}).get("Tools", "")
    tools = [t.split(".", 1)[-1].strip() for t in tools_raw.split("\n") if t.strip()]
    for t in tools:
        tool_categories[normalize_tool(t)] += 1


# ============================================================
# 4. GROUP ATTACHMENTS
# ============================================================

pdf = sum(attachment_types[e] for e in [".pdf"])
excel = sum(attachment_types[e] for e in [".csv", ".xlsx"])
images = sum(attachment_types[e] for e in [".png", ".jpg", ".jpeg"])
audio_video = sum(attachment_types[e] for e in [".mp3", ".m4a", ".mp4"])
xml_json = sum(attachment_types[e] for e in [".xml", ".json", ".jsonld"])


# Helper for percentage
def pct(x):
    return f"{(100*x/total_q):.1f}\\%" if total_q else "--"


# ============================================================
# 5. LATEX OUTPUT (COMPACT TABLE)
# ============================================================

latex = f"""
\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabular}}{{ccc}}
\\toprule
\\textbf{{Dataset Summary}} & \\textbf{{Attachment Types}} & \\textbf{{Tool Categories}} \\\\
\\midrule

% =======================
% Column 1: Summary
% =======================
\\begin{{tabular}}{{lrr}}
\\toprule
Statistic & Count & \% \\\\
\\midrule
Total Questions & {total_q} & 100\\% \\\\
With Attachments & {total_attachments} & {pct(total_attachments)} \\\\
Level~1 & {level_dist[1]} & {pct(level_dist[1])} \\\\
Level~2 & {level_dist[2]} & {pct(level_dist[2])} \\\\
Level~3 & {level_dist[3]} & {pct(level_dist[3])} \\\\
\\bottomrule
\\end{{tabular}}
&
% =======================
% Column 2: Attachments
% =======================
\\begin{{tabular}}{{lrr}}
\\toprule
Type & Count & \% \\\\
\\midrule
PDF & {pdf} & {pct(pdf)} \\\\
Excel/CSV & {excel} & {pct(excel)} \\\\
Images & {images} & {pct(images)} \\\\
Audio/Video & {audio_video} & {pct(audio_video)} \\\\
XML/JSON & {xml_json} & {pct(xml_json)} \\\\
\\bottomrule
\\end{{tabular}}
&
% =======================
% Column 3: Tools
% =======================
\\begin{{tabular}}{{lrr}}
\\toprule
Category & Count & \% \\\\
\\midrule
Web Browsing & {tool_categories['Web Browsing']} & {pct(tool_categories['Web Browsing'])} \\\\
Search Engine & {tool_categories['Search Engine']} & {pct(tool_categories['Search Engine'])} \\\\
Coding & {tool_categories['Coding']} & {pct(tool_categories['Coding'])} \\\\
File Reading & {tool_categories['File Reading']} & {pct(tool_categories['File Reading'])} \\\\
Multi-Modality & {tool_categories['Multi-Modality']} & {pct(tool_categories['Multi-Modality'])} \\\\
\\bottomrule
\\end{{tabular}}

\\\\
\\bottomrule
\\end{{tabular}}

\\caption{{Statistics of the GAIA validation split (2023\\_all).}}
\\label{{tab:gaia-stats}}
\\end{{table}}
"""

print(latex)
print("\n✔ LaTeX table generated.\n")
