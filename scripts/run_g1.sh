#!/bin/bash
#SBATCH --account=digital_human
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=24G

source activate /work/courses/digital_human/team10/jjanaskar/conda_envs/isaaclab
module add cuda/13.0
cd /work/courses/digital_human/team10/jjanaskar/code/MimicKit
python mimickit/run.py --arg_file args/view_motion_g1_args.txt --engine_config data/engines/isaac_lab_engine.yaml --mode test --video true --visualize false --test_episodes 1 --logger tb


#python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --mode test --video true --visualize false --test_episodes 1 --logger tb --num_envs 4
