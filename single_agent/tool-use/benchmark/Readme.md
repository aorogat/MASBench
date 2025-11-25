# GAIA Benchmark (Validation Split) — Dataset Processing, Files, and Statistics

This directory contains utilities for downloading, analyzing, and preparing the GAIA benchmark (2023 release) for tool-use and multi-agent experimentation.

GAIA is a dataset designed to evaluate next-generation AI agents equipped with capabilities such as tool-use, web browsing, multimodal perception, file processing, and coding. Each question in GAIA includes human-provided metadata specifying the *capabilities* and optional *tools* needed to solve it.

We focus on the **validation split** of the `2023_all` subset, which contains full metadata including Levels 1–3 and file attachments.

---

## 📦 Downloading the GAIA Dataset

The dataset is fetched from HuggingFace using:

```python
from huggingface_hub import snapshot_download
data_dir = snapshot_download("gaia-benchmark/GAIA", repo_type="dataset")
```

HuggingFace stores datasets in the standard HuggingFace cache directory:

```
~/.cache/huggingface/hub/datasets--gaia-benchmark--GAIA/snapshots/<hash>/
```

This path is automatically managed by HuggingFace and is portable across machines. No custom paths are required.

The directory includes:

```
2023/
  validation/
    *.pdf, *.xlsx, *.csv, *.png, *.jpg, *.xml, *.json, *.mp3, *.mp4, ...
  test/
metadata files
README.md
```

All attachments referenced by GAIA are included locally after download.

---

## 📁 Listing Attachment Files

Once downloaded, all GAIA attachments can be listed using standard filesystem commands:

**List all validation files:**
```bash
ls <data_dir>/2023/validation
```

**Only PDFs:**
```bash
ls <data_dir>/2023/validation/*.pdf
```

**Only Excel/CSV:**
```bash
ls <data_dir>/2023/validation/*.xlsx
ls <data_dir>/2023/validation/*.csv
```

**Images:**
```bash
ls <data_dir>/2023/validation/*.png
ls <data_dir>/2023/validation/*.jpg
```

**Audio / Video:**
```bash
ls <data_dir>/2023/validation/*.mp3
ls <data_dir>/2023/validation/*.m4a
ls <data_dir>/2023/validation/*.mp4
```

GAIA ships all attachments directly; no external URLs or downloads are needed.

---

## 📊 Dataset Statistics (Validation Split)

The following table summarizes the 165 questions in the GAIA validation split (2023_all), including level distribution, attachment types, and tool capability categories extracted from annotator metadata.

| **Dataset Summary** | Count | % |
|---|---|---|
| Total | 165 | 100% |
| Attachments | 38 | 23.0% |
| Level 1 | 53 | 32.1% |
| Level 2 | 86 | 52.1% |
| Level 3 | 26 | 15.8% |

| **Attachment Types** | Count | % |
|---|---|---|
| PDF | 3 | 1.8% |
| Excel/CSV | 14 | 8.5% |
| Images | 10 | 6.1% |
| Audio/Video | 3 | 1.8% |
| XML/JSON | 1 | 0.6% |

| **Tool Categories** | Count | % |
|---|---|---|
| Web Browsing | 124 | 75.2% |
| Search Engine | 112 | 67.9% |
| Coding | 52 | 31.5% |
| File Reading | 41 | 24.8% |
| Multi-Modality | 43 | 26.1% |

---

## 🧠 Usage in Multi-Agent Framework Experiments

GAIA exercises five capability categories:

1. **Web browsing** — links, maps, online content
2. **Search engines** — query-needed tasks
3. **File reading** — PDF, Excel, CSV, XML
4. **Coding** — calculations, scripts, computation
5. **Multi-modality** — images, audio, video, OCR

These capabilities directly map to tool availability in multi-agent frameworks such as LangGraph, CrewAI, and OpenAI Agents. The statistics above can be used to:

- Design controlled tool-use experiments
- Scale tool availability
- Benchmark reasoning performance under different toolsets
- Evaluate framework overhead and orchestration complexity

---

## ▶️ Running the Statistics Script

Use:

```bash
python gaia_stats_validation.py
```

This script will:

- Load GAIA
- Compute dataset statistics
- Extract tool categories
- Generate the compact LaTeX table

Output LaTeX is printed directly to terminal.

---

## 📬 Contact

For issues or contributions, please open a GitHub issue or pull request.