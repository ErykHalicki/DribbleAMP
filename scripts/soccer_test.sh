#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/MimicKit"

MODEL_FILE="${MODEL_FILE:-$SCRIPT_DIR/output/model.pt}"

# MODE presets pick matching env + agent configs (override individually with
# ENV_CONFIG / AGENT_CONFIG if needed):
#   phase1 (default) -> locomotion env, base AMP agent
#   phase2           -> dribbling env, phase-2 AMP agent
#   no_amp           -> dribbling env, no-AMP agent (pure task reward ablation)
MODE="${MODE:-phase1}"
case "$MODE" in
    phase1) DEF_ENV=amp_soccer_humanoid_env_phase1.yaml; DEF_AGENT=amp_task_humanoid_agent.yaml ;;
    phase2) DEF_ENV=amp_soccer_humanoid_env_phase2.yaml; DEF_AGENT=amp_task_humanoid_agent_phase2.yaml ;;
    no_amp) DEF_ENV=amp_soccer_humanoid_env_phase2.yaml; DEF_AGENT=amp_task_humanoid_agent_phase2_no_amp.yaml ;;
    *) echo "ERROR: unknown MODE '$MODE' (expected 'phase1', 'phase2', or 'no_amp')" >&2; exit 1 ;;
esac

ENV_CONFIG="${ENV_CONFIG:-data/envs/$DEF_ENV}"
AGENT_CONFIG="${AGENT_CONFIG:-data/agents/$DEF_AGENT}"

"$SCRIPT_DIR/.venv/bin/python" mimickit/run.py --mode test --devices cpu \
  --engine_config data/engines/newton_engine.yaml \
  --env_config "$ENV_CONFIG" \
  --agent_config "$AGENT_CONFIG" \
  --model_file "$MODEL_FILE" \
  "$@"
