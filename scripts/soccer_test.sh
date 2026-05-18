#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/MimicKit"

MODEL_FILE="${MODEL_FILE:-$SCRIPT_DIR/output/model.pt}"

"$SCRIPT_DIR/.venv/bin/python" mimickit/run.py --mode test --devices cpu \
  --engine_config data/engines/newton_engine.yaml \
  --env_config data/envs/amp_soccer_humanoid_env_phase1.yaml \
  --agent_config data/agents/amp_task_humanoid_agent.yaml \
  --model_file "$MODEL_FILE" \
  "$@"
