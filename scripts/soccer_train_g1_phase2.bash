#!/bin/bash
#SBATCH --account=digital_human_jobs
#SBATCH --time=08:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=24G

# Resume from phase-1 checkpoint with higher ball-velocity weight.
# Override checkpoint: sbatch --export=ALL,PHASE1_MODEL=output/soccer_g1_phase1/model_xxxxx.pt scripts/soccer_train_g1_phase2.bash

set -euo pipefail

PROJECT_DIR=/work/courses/digital_human/team10/DribbleAMP

source /home/jjanaskar/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/digital_human/team10/jjanaskar/conda_envs/isaaclab
module add cuda/13.0

cd "${PROJECT_DIR}/MimicKit"

# Resolve PHASE1_MODEL: explicit > latest phase 1 final model.pt > latest phase 1 int_models checkpoint.
if [ -z "${PHASE1_MODEL:-}" ]; then
    PHASE1_FINAL=$(ls -1 output/soccer_g1_phase1*/model.pt 2>/dev/null | sort | tail -n 1 || true)
    PHASE1_INT=$(ls -1 output/soccer_g1_phase1*/int_models/model_*.pt 2>/dev/null | sort | tail -n 1 || true)
    if [ -n "${PHASE1_FINAL}" ]; then
        PHASE1_MODEL="${PHASE1_FINAL}"
    elif [ -n "${PHASE1_INT}" ]; then
        PHASE1_MODEL="${PHASE1_INT}"
    else
        echo "ERROR: no phase 1 checkpoint found under output/soccer_g1_phase1*/; set PHASE1_MODEL explicitly." >&2
        exit 1
    fi
fi
echo "Phase 1 source model: ${PHASE1_MODEL}"

RUN_PREFIX=output/soccer_g1_phase2
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
    echo "Resuming phase 2 from intermediate checkpoint ${RESUME_MODEL} (out_dir=${OUT_DIR})"
elif [ -n "${LATEST_FINAL}" ]; then
    OUT_DIR=$(dirname "${LATEST_FINAL}")/
    RESUME_MODEL="${LATEST_FINAL}"
    echo "Resuming phase 2 from ${RESUME_MODEL} (out_dir=${OUT_DIR})"
else
    OUT_DIR=${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)/
    RESUME_MODEL="${PHASE1_MODEL}"
    echo "Starting phase 2 from phase 1 checkpoint ${RESUME_MODEL} (out_dir=${OUT_DIR})"
fi

python -u mimickit/run.py --mode train --num_envs 2048 --engine_config data/engines/isaac_lab_engine.yaml --env_config data/envs/amp_soccer_g1_env_phase2.yaml --agent_config data/agents/amp_task_g1_agent_phase2.yaml --model_file "${RESUME_MODEL}" --visualize false --out_dir "${OUT_DIR}" --logger tb --save_int_models true
