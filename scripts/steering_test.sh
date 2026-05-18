#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/MimicKit"

"$SCRIPT_DIR/.venv/bin/python" mimickit/run.py --mode test --devices cpu \
  --engine_config data/engines/newton_engine.yaml \
  --env_config data/envs/amp_steering_humanoid_env.yaml \
  --agent_config data/agents/amp_task_humanoid_agent.yaml \
  --model_file "$SCRIPT_DIR/output/model.pt" \
  "$@"
