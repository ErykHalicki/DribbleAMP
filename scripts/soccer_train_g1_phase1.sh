#!/bin/bash
#SBATCH --account=digital_human_jobs
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G

# Setup Conda environment (same as our test script)
source /home/jjanaskar/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/digital_human/team10/jjanaskar/conda_envs/isaaclab
module add cuda/13.0

PROJECT_DIR="/work/courses/digital_human/team10/DribbleAMP"
cd "${PROJECT_DIR}/MimicKit"

OUT_DIR=output/soccer_g1_phase1/
RESUME_ARGS=""

# Auto-resume logic
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

# Training command. Notice I used Isaac Lab here for training speed (allows for large `num_envs`), 
# but you can easily change `--engine_config` to newton if you prefer.
python -u mimickit/run.py --mode train --num_envs 2048 \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_soccer_g1_env_phase1.yaml \
  --agent_config data/agents/amp_task_g1_agent.yaml \
  --visualize false --out_dir "${OUT_DIR}" \
  --logger tb --save_int_models true ${RESUME_ARGS} "$@"