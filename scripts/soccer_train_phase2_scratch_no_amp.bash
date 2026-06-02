#!/bin/bash
#SBATCH --account=digital_human_jobs
#SBATCH --time=24:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --mem=24G

# No-AMP ablation. Identical setup to the phase-2-from-scratch run
# (soccer_train_phase2.bash with FROM_SCRATCH=1), but uses the no-AMP agent
# config (disc_reward_weight=0, task_reward_weight=1): pure task reward, no
# adversarial style prior. Always trains from scratch.
# Force a brand-new run instead of resuming: sbatch --export=ALL,FRESH=1 scripts/soccer_train_phase2_scratch_no_amp.bash

set -euo pipefail

PROJECT_DIR=/work/courses/digital_human/team10/DribbleAMP
VENV_DIR=${PROJECT_DIR}/.venv

module add cuda/13.0

cd "${PROJECT_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
    uv venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

if [ -f requirements.txt ]; then
    uv pip install -r requirements.txt
fi
if [ -f MimicKit/requirements.txt ]; then
    uv pip install -r MimicKit/requirements.txt
fi

cd "${PROJECT_DIR}/MimicKit"

RUN_PREFIX=output/soccer_phase2_scratch_no_amp
FRESH="${FRESH:-0}"
if [ "${FRESH}" = "1" ]; then
    LATEST_INT=""
    LATEST_FINAL=""
else
    LATEST_INT=$(ls -1 ${RUN_PREFIX}*/int_models/model_*.pt 2>/dev/null | sort | tail -n 1 || true)
    LATEST_FINAL=$(ls -1 ${RUN_PREFIX}*/model.pt 2>/dev/null | sort | tail -n 1 || true)
fi
if [ -n "${LATEST_INT}" ]; then
    OUT_DIR=$(dirname "$(dirname "${LATEST_INT}")")/
    RESUME_MODEL="${LATEST_INT}"
    echo "Resuming no-AMP run from intermediate checkpoint ${RESUME_MODEL} (out_dir=${OUT_DIR})"
elif [ -n "${LATEST_FINAL}" ]; then
    OUT_DIR=$(dirname "${LATEST_FINAL}")/
    RESUME_MODEL="${LATEST_FINAL}"
    echo "Resuming no-AMP run from ${RESUME_MODEL} (out_dir=${OUT_DIR})"
else
    OUT_DIR=${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)/
    RESUME_MODEL=""
    echo "Starting no-AMP run from scratch (out_dir=${OUT_DIR})"
fi

MODEL_ARG=()
if [ -n "${RESUME_MODEL}" ]; then
    MODEL_ARG=(--model_file "${RESUME_MODEL}")
fi

python -u mimickit/run.py --mode train --num_envs 1024 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_soccer_humanoid_env_phase2.yaml --agent_config data/agents/amp_task_humanoid_agent_phase2_no_amp.yaml "${MODEL_ARG[@]}" --visualize false --out_dir "${OUT_DIR}" --logger tb --save_int_models true
