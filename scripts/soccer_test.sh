#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR/MimicKit"

# MODE presets pick matching env + agent configs (override individually with
# ENV_CONFIG / AGENT_CONFIG if needed):
#   phase1 (default) -> humanoid chase env, base AMP agent
#   phase2           -> humanoid dribbling env, phase-2 AMP agent
#   no_amp           -> humanoid dribbling env, no-AMP agent (pure task reward ablation)
#   g1_phase1        -> Unitree G1 chase env, G1 phase-1 AMP agent
#   g1_phase2        -> Unitree G1 dribbling env, G1 phase-2 AMP agent
MODE="${MODE:-phase1}"
case "$MODE" in
    phase1)    DEF_ENV=amp_soccer_humanoid_env_phase1.yaml; DEF_AGENT=amp_task_humanoid_agent.yaml;              DEF_MODEL=humanoid_stage1.pt ;;
    phase2)    DEF_ENV=amp_soccer_humanoid_env_phase2.yaml; DEF_AGENT=amp_task_humanoid_agent_phase2.yaml;       DEF_MODEL=humanoid_stage1_stage2.pt ;;
    no_amp)    DEF_ENV=amp_soccer_humanoid_env_phase2.yaml; DEF_AGENT=amp_task_humanoid_agent_phase2_no_amp.yaml; DEF_MODEL=humanoid_no_amp.pt ;;
    g1_phase1) DEF_ENV=amp_soccer_g1_env_phase1.yaml;       DEF_AGENT=amp_task_g1_agent_phase1.yaml;             DEF_MODEL=g1_stage1.pt ;;
    g1_phase2) DEF_ENV=amp_soccer_g1_env_phase2.yaml;       DEF_AGENT=amp_task_g1_agent_phase2.yaml;             DEF_MODEL=g1_stage2.pt ;;
    *) echo "ERROR: unknown MODE '$MODE' (expected 'phase1', 'phase2', 'no_amp', 'g1_phase1', or 'g1_phase2')" >&2; exit 1 ;;
esac

# Default model: the pretrained checkpoint shipped under models/ for this MODE,
# falling back to output/model.pt (the sync_model.bash landing spot).
if [ -z "${MODEL_FILE:-}" ]; then
    if [ -f "$SCRIPT_DIR/models/$DEF_MODEL" ]; then
        MODEL_FILE="$SCRIPT_DIR/models/$DEF_MODEL"
    else
        MODEL_FILE="$SCRIPT_DIR/output/model.pt"
    fi
fi

ENV_CONFIG="${ENV_CONFIG:-data/envs/$DEF_ENV}"
AGENT_CONFIG="${AGENT_CONFIG:-data/agents/$DEF_AGENT}"

"$SCRIPT_DIR/.venv/bin/python" mimickit/run.py --mode test --devices cpu \
  --engine_config data/engines/newton_engine.yaml \
  --env_config "$ENV_CONFIG" \
  --agent_config "$AGENT_CONFIG" \
  --model_file "$MODEL_FILE" \
  "$@"
