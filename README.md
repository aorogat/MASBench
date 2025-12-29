# Multi-Agent Framework Experiments

[![Python 3.12.3](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/)
[![Dependencies](https://img.shields.io/badge/dependencies-pinned-green.svg)](requirements.lock)
[![Research](https://img.shields.io/badge/status-research-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive experimental framework for systematic evaluation of large language model (LLM)-based single-agent and multi-agent systems across multiple dimensions: memory architectures, reasoning strategies, tool integration, coordination protocols, specialization patterns, and framework overhead analysis.

## Overview

This repository supports rigorous, reproducible experimentation where each study is self-contained and independently documented. The modular architecture enables researchers to focus on specific aspects of agent behavior without requiring understanding of the entire codebase.

---

## 📁 Repository Structure

```
.
├── single_agent/          # Single-agent capability evaluations
├── multi_agent/           # Multi-agent coordination experiments
├── benchmarks/            # Benchmark definitions and evaluation logic
├── llms/                  # LLM backend abstractions and adapters
├── results/               # Experimental outputs and analysis artifacts
├── scripts/               # Execution entry points and utilities
├── data/                  # Benchmark datasets
├── logs/                  # Runtime logs (not required for reproduction)
├── requirements.lock      # Pinned dependency versions
└── README.md             # This file
```

---

## 🔬 Experimental Design Philosophy

Each experiment is organized as a **standalone module** containing:

- **Clear scope definition** – Precise research questions and hypotheses
- **Isolated execution scripts** – Self-contained runners with minimal dependencies
- **Local documentation** – Dedicated `README.md` with:
  - Task specifications
  - Framework configurations
  - Reproduction instructions
  - Result interpretation guidelines

> **Navigation:** Explore experiments through their local README files rather than treating this as a monolithic codebase.

---

## 🧪 Single-Agent Experiments

**Location:** `single_agent/`

### Available Studies

#### Memory Evaluation
**Path:** `single_agent/memory/`

Comprehensive analysis of memory architectures using MemoryAgentBench:
- Long-term memory with vector database retrieval
- Short-term accumulation-based memory
- Hybrid memory configurations
- Cross-framework memory behavior comparison

#### Framework Overhead Analysis
**Path:** `single_agent/framework_overhead/`

Quantifies runtime and orchestration costs introduced by agent frameworks, isolated from task complexity variables.

#### Reasoning & Planning
**Path:** `single_agent/reasoning/`

Evaluates reasoning accuracy, failure mode analysis, and computational efficiency across:
- GSM8K (mathematical reasoning)
- CSQA (commonsense reasoning)
- Multiple reasoning strategies (CoT, ReAct, self-consistency)

#### Specialization & Role Prompting
**Path:** `single_agent/specialization/`

Investigates the impact of role assignment and expert augmentation on:
- Predictive task performance
- Analytical reasoning quality
- Domain-specific knowledge application

#### Tool Use Integration
**Path:** `single_agent/tool_use/`

Benchmarks agent-tool interaction patterns using StableToolBench:
- Tool selection strategies
- Error recovery mechanisms
- Multi-tool coordination

---

## 🤖 Multi-Agent Experiments

**Location:** `multi_agent/`

### Research Focus

- Communication topology design and analysis
- Consensus formation mechanisms
- Leader election protocols
- Scalability characteristics
- Framework-specific orchestration patterns

### Primary Entry Point

**Topology & Coordination:** `multi_agent/topology/`

Includes visualization utilities, statistical analysis scripts, and framework-specific execution runners.

---

## 📊 Results & Artifacts

**Location:** `results/`

Contains:
- Per-framework result directories
- Per-configuration experimental outputs
- LaTeX-formatted tables for publication
- Analytical figures (memory curves, coordination efficiency, etc.)

Raw result files are preserved for **transparency and reproducibility** but are not required to understand or execute experiments.

---

## ⚙️ Environment & Reproducibility

### Requirements

- **Python Version:** `3.12.3`
- **Dependencies:** Exact library versions specified in `requirements.lock`

### Setup Instructions

1. **Environment Configuration:**
   ```bash
   # See session_setup.md for detailed instructions
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.lock
   ```

2. **Tmux Workflows** (optional):
   See `tmux.md` for distributed execution patterns

All experiments are designed for **full reproducibility** from a clean Python environment.

---

## 🚀 Getting Started

### Quick Start Guide

1. **Choose an experiment:**
   - Memory systems → `single_agent/memory/`
   - Agent coordination → `multi_agent/topology/`
   - Tool integration → `single_agent/tool_use/`

2. **Read the local README:**
   Each experiment folder contains detailed setup and execution instructions

3. **Run the provided scripts:**
   ```bash
   cd single_agent/memory
   python run_experiment.py --config default.yaml
   ```

4. **Inspect results:**
   ```bash
   ls ../../results/memory/
   ```

---

## 📄 Citation & Usage

This repository supports ongoing research in multi-agent systems, memory architectures, and LLM evaluation methodologies.

If you use or extend this codebase, please cite the corresponding papers listed in the experiment-specific README files.

### License

[Specify your license here]

---

## 🤝 Contributing

Contributions are welcome. Please:
- Maintain the modular structure
- Include standalone README files for new experiments
- Ensure reproducibility with pinned dependencies
- Follow existing code organization patterns

---

## 📧 Contact

[Your contact information or links to project maintainers]

---

**Note:** This repository is intentionally modular. You do not need to understand the entire codebase to reproduce or extend a specific study. Each experiment is designed to stand on its own.