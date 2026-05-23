#!/bin/bash
#SBATCH --account=digital_human
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=64G

source /home/jjanaskar/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/digital_human/team10/jjanaskar/conda_envs/isaaclab
module add cuda/13.0

SCRIPT_DIR="/work/courses/digital_human/team10/DribbleAMP"
cd "$SCRIPT_DIR/MimicKit"

MODEL_FILE="${MODEL_FILE:-$SCRIPT_DIR/MimicKit/output/soccer_g1_phase1/model.pt}"
TEST_EPISODES="${TEST_EPISODES:-8}"
OUT_DIR="${OUT_DIR:-output/soccer_test_g1}"

PYTHONUNBUFFERED=1 python -u mimickit/run.py --mode test \
  --engine_config data/engines/isaac_lab_engine.yaml \
  --env_config data/envs/amp_soccer_g1_env_phase1.yaml \
  --agent_config data/agents/amp_task_g1_agent.yaml \
  --model_file "$MODEL_FILE" \
  --test_episodes "$TEST_EPISODES" \
  --out_dir "$OUT_DIR" \
  --visualize false \
  --video true \
  "$@"