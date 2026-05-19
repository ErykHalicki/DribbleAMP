#!/bin/bash
#SBATCH --account=digital_human_jobs
#SBATCH --time=48:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=24G

# Resume from phase-1 checkpoint with higher ball-velocity weight.
# Override checkpoint: sbatch --export=ALL,PHASE1_MODEL=output/soccer_phase1/model_xxxxx.pt scripts/soccer_train_phase2.bash

set -euo pipefail

PROJECT_DIR=/work/courses/digital_human/team10/DribbleAMP
VENV_DIR=${PROJECT_DIR}/.venv
PHASE1_MODEL="${PHASE1_MODEL:-output/soccer_phase1/model.pt}"

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

OUT_DIR=output/soccer_phase2/
LATEST_INT=$(ls -1 "${OUT_DIR}int_models/"model_*.pt 2>/dev/null | sort | tail -n 1 || true)
if [ -n "${LATEST_INT}" ]; then
    RESUME_MODEL="${LATEST_INT}"
    echo "Resuming phase 2 from intermediate checkpoint ${RESUME_MODEL}"
elif [ -f "${OUT_DIR}model.pt" ]; then
    RESUME_MODEL="${OUT_DIR}model.pt"
    echo "Resuming phase 2 from ${RESUME_MODEL}"
else
    RESUME_MODEL="${PHASE1_MODEL}"
    echo "Starting phase 2 from phase 1 checkpoint ${RESUME_MODEL}"
fi

python -u mimickit/run.py --mode train --num_envs 1024 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_soccer_humanoid_env_phase2.yaml --agent_config data/agents/amp_task_humanoid_agent.yaml --model_file "${RESUME_MODEL}" --visualize false --out_dir "${OUT_DIR}" --logger tb --save_int_models true
