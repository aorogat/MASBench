#!/usr/bin/env bash
set -euo pipefail

echo "📂 Creating project structure for maf-experiments..."

# Main directories
mkdir -p \
  single-agent/tool-use \
  single-agent/memory \
  single-agent/reasoning \
  multi-agent/communication \
  multi-agent/coordination \
  multi-agent/environment \
  multi-agent/topology \
  configs \
  data \
  logs \
  models \
  results \
  scripts

# Function to create test files inside a subfolder
create_experiment_files() {
  local folder=$1
  echo "  ↳ Creating experiment files in $folder"
  cat > $folder/langgraph_test.py << 'PY'
print("Running LangGraph experiment for $(basename $(dirname $0))/$(basename $0)")
PY
  cat > $folder/crewai_test.py << 'PY'
print("Running CrewAI experiment for $(basename $(dirname $0))/$(basename $0)")
PY
  cat > $folder/concordia_test.py << 'PY'
print("Running Concordia experiment for $(basename $(dirname $0))/$(basename $0)")
PY
  cat > $folder/README.md << 'MD'
# Experiment: $(basename $folder)

This folder contains experiments for the **$(basename $folder)** setup.

- **langgraph_test.py** → run with LangGraph
- **crewai_test.py** → run with CrewAI
- **concordia_test.py** → run with Concordia

Extend these scripts with the actual experiment logic.
MD
}

# Iterate over all experiment subfolders
for sub in single-agent/tool-use single-agent/memory single-agent/reasoning \
           multi-agent/communication multi-agent/coordination multi-agent/environment multi-agent/topology
do
  create_experiment_files $sub
done

# Requirements
cat > requirements.txt << 'REQ'
langgraph
crewai
gdm-concordia
REQ

# .gitignore
cat > .gitignore << 'IGN'
mafenv/
__pycache__/
*.pyc
.env
logs/
results/
IGN

# Placeholder runner scripts
cat > scripts/run_single_agent_tool_use.py << 'PY'
print("Run: single-agent tool-use experiment (placeholder).")
PY

cat > scripts/run_multi_agent_coordination.py << 'PY'
print("Run: multi-agent coordination experiment (placeholder).")
PY

echo "✅ Project structure created."
echo "Next steps:"
echo "  python3 -m venv mafenv && source mafenv/bin/activate"
echo "  pip install -U pip"
echo "  pip install -r requirements.txt || (sed -i 's/gdm-concordia/concordia/' requirements.txt && pip install -r requirements.txt)"
