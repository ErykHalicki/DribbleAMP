#!/bin/bash
#SBATCH --account=digital_human_jobs
#SBATCH --time=48:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=24G

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

RUN_PREFIX=output/soccer_phase1
RESUME_ARGS=""
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
    RESUME_ARGS="--model_file ${LATEST_INT}"
    echo "Resuming from ${LATEST_INT} (out_dir=${OUT_DIR})"
elif [ -n "${LATEST_FINAL}" ]; then
    OUT_DIR=$(dirname "${LATEST_FINAL}")/
    RESUME_ARGS="--model_file ${LATEST_FINAL}"
    echo "Resuming from ${LATEST_FINAL} (out_dir=${OUT_DIR})"
else
    OUT_DIR=${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)/
    echo "No checkpoint found, starting fresh (out_dir=${OUT_DIR})"
fi

python -u mimickit/run.py --mode train --num_envs 1024 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_soccer_humanoid_env_phase1.yaml --agent_config data/agents/amp_task_humanoid_agent.yaml --visualize false --out_dir "${OUT_DIR}" --logger tb --save_int_models true ${RESUME_ARGS}
