#!/bin/bash
#SBATCH --account=digital_human
#SBATCH --time=00:05:00
#SBATCH --gpus=gb10:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G

echo "=== HOST ==="
hostname
echo "=== ARCH ==="
uname -m
echo "=== GPU ==="
nvidia-smi
echo "=== CONDA ENV TEST ==="
source /home/jjanaskar/miniconda3/etc/profile.d/conda.sh
conda activate /work/courses/digital_human/team10/jjanaskar/conda_envs/isaaclab 2>&1
echo "Python: $(which python)"
echo "Python arch: $(python -c 'import platform; print(platform.machine())' 2>&1)"
echo "=== TORCH IMPORT ==="
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())" 2>&1
echo "=== ISAAC IMPORT ==="
python -c "import isaaclab; print('isaaclab ok')" 2>&1 || echo "isaaclab import failed"
