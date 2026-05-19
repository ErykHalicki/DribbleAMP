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

OUT_DIR=output/soccer_phase1/
RESUME_ARGS=""
LATEST_INT=$(ls -1 "${OUT_DIR}int_models/"model_*.pt 2>/dev/null | sort | tail -n 1 || true)
if [ -n "${LATEST_INT}" ]; then
    RESUME_ARGS="--model_file ${LATEST_INT}"
    echo "Resuming from ${LATEST_INT}"
elif [ -f "${OUT_DIR}model.pt" ]; then
    RESUME_ARGS="--model_file ${OUT_DIR}model.pt"
    echo "Resuming from ${OUT_DIR}model.pt"
else
    echo "No checkpoint found, starting fresh"
fi

python -u mimickit/run.py --mode train --num_envs 512 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_soccer_humanoid_env_phase1.yaml --agent_config data/agents/amp_task_humanoid_agent.yaml --visualize false --out_dir "${OUT_DIR}" --logger tb --save_int_models true ${RESUME_ARGS}
