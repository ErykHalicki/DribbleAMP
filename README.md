# DribbleAMP

Velocity-conditioned football-dribbling policies for humanoid robots,
trained with [Adversarial Motion Priors (AMP)](https://arxiv.org/abs/2104.02180)
on top of [MimicKit](https://github.com/xbpeng/MimicKit). Our task reward is
adapted from [DribbleMaster](https://zhuoheng0910.github.io/dribble-master/);
the style reward comes from an AMP discriminator instead of their hand-crafted
penalties.

Course project for ETH Z&uuml;rich's Digital Humans (Eryk Halicki, Jay Janaskar).

## Demos

- **Stage 1 (chase + kick)** — Unitree G1 learns to approach the ball:
  https://youtu.be/f0HXS8HKP3Q
- **Stage 2 (dribble)** — G1 follows a commanded ball velocity:
  https://youtu.be/H7UHbeMz4UI

In the videos, the **red arrow** is the commanded ball velocity and the
**green arrow** is the actual ball velocity.

## Method (TL;DR)

Per-step reward:

```
r = 0.8 · r_task + 0.2 · r_style
r_task = w_s · speed_match + w_d · direction_match + w_r · distance_to_ball
```

Two-stage curriculum:

| Stage | Goal | Weights (w_s, w_d, w_r) | Ball spawn |
|-------|------|--------------------------|------------|
| 1 (chase)   | balance + approach + kick | (0.6, 0.0, 0.4) | 5–10 m |
| 2 (dribble) | track ball velocity        | (0.3, 0.5, 0.2) | 1–3 m  |

The AMP discriminator is trained on the locomotion clips (walk + run) shipped
with MimicKit — no dribbling reference motion is needed.


## Setup

Tested with **Python 3.11** and **CUDA 13** on a Linux GPU box.

```bash
git clone https://github.com/ErykHalicki/DribbleAMP.git
cd DribbleAMP

# Create an env (any of: venv, conda, mamba; example with venv)
python -m venv .venv
source .venv/bin/activate

# Install Python deps
pip install -r requirements.txt
pip install -r MimicKit/requirements.txt

# Newton (Warp-based physics) + Isaac Lab — follow upstream install guides
#   Newton: https://github.com/NVIDIA/Newton
#   Isaac Lab: https://isaac-sim.github.io/IsaacLab/
```

### SLURM cluster

The bash scripts under `scripts/` are written for the ETH student cluster.
They activate a shared conda env and load a CUDA module:

```bash
source /home/<your-user>/miniconda3/etc/profile.d/conda.sh
conda activate <path-to-your-env>
module add cuda/13.0
```

Edit these lines in each script to point at your own conda env, or replace
them with `source .venv/bin/activate` if you used a venv (Newton scripts only; Isaac Lab needs conda because of Omniverse).

## Running the paper demos (pretrained models)

The checkpoints behind every result in the paper are included in the repo
under `models/`:

| File | Policy | Paper result |
|------|--------|--------------|
| `humanoid_stage1.pt`        | Humanoid, Stage 1 only (chase + kick) | curriculum ablation |
| `humanoid_stage1_stage2.pt` | Humanoid, Stage 1 + Stage 2 (ours)    | main result |
| `humanoid_stage2_only.pt`   | Humanoid, Stage 2 from scratch        | curriculum ablation |
| `humanoid_no_amp.pt`        | Humanoid, task reward only            | AMP ablation |
| `g1_stage1.pt`              | Unitree G1, Stage 1                   | G1 chase demo |
| `g1_stage2.pt`              | Unitree G1, Stage 2                   | G1 dribble demo |

`scripts/soccer_test.sh` opens an interactive Newton viewer (CPU, no GPU or
cluster needed — just the venv from [Setup](#setup)). Each `MODE` preset picks
the matching env + agent configs and defaults to the corresponding checkpoint
above:

```bash
# Humanoid, full two-stage curriculum (our main result)
MODE=phase2 scripts/soccer_test.sh

# Humanoid ablations
MODE=phase1 scripts/soccer_test.sh   # Stage 1 only: approaches + kicks, can't steer
MODE=no_amp scripts/soccer_test.sh   # no style prior: dribbles well, unnatural gait

# Unitree G1
MODE=g1_phase1 scripts/soccer_test.sh
MODE=g1_phase2 scripts/soccer_test.sh
```

To run a different checkpoint with the same configs, pass `MODEL_FILE`
explicitly — e.g. the Stage-2-only ablation reuses the `phase2` configs:

```bash
MODE=phase2 MODEL_FILE=$PWD/models/humanoid_stage2_only.pt scripts/soccer_test.sh
```

To re-fetch all of the above from the training cluster (team members only;
prompts once for your cluster password):

```bash
scripts/download_paper_models.bash
```

## Training

All training is launched via SLURM `sbatch` on the cluster. Each script
auto-resumes from the latest checkpoint in its output directory; pass
`FRESH=1` to start from scratch.

### G1 (Newton)

```bash
# Stage 1: chase + kick
sbatch scripts/soccer_train_g1_phase1_newton.bash

# Stage 2: dribble (auto-loads the latest Stage-1 checkpoint)
sbatch scripts/soccer_train_g1_phase2_newton.bash
```

### Humanoid (Newton)

```bash
sbatch scripts/soccer_train_phase1.bash
sbatch scripts/soccer_train_phase2.bash
```

Outputs go to `MimicKit/output/soccer_<...>_phase<N>_<timestamp>/`.
Intermediate model checkpoints are saved every 200 iters under `int_models/`.

## Rendering test videos

Pass an explicit `MODEL_FILE` and `OUT_DIR`; videos render at 1080p
(`TEST_EPISODES=2` is the safe cap on a 24 GB GPU).

```bash
sbatch \
  --export=ALL,\
MODEL_FILE=$PWD/MimicKit/output/soccer_g1_newton_phase2_*/int_models/model_0000005000.pt,\
OUT_DIR=output/test_g1_phase2_iter5000,TEST_EPISODES=2 \
  scripts/soccer_test_g1_newton.sh \
  --env_config data/envs/amp_soccer_g1_env_phase2.yaml \
  --agent_config data/agents/amp_task_g1_agent_phase2.yaml
```

Output mp4 lands at `MimicKit/<OUT_DIR>/test_video.mp4`.

## Repository layout

```
MimicKit/                       # vendored fork of MimicKit
  data/envs/amp_soccer_*.yaml   # task configs (humanoid + G1, phase 1 + 2)
  data/agents/amp_task_*.yaml   # AMP agent configs
  data/datasets/dataset_g1_locomotion.yaml  # AMP demo motion list
  mimickit/envs/task_soccer_env.py          # soccer task + reward
  mimickit/engines/newton_engine.py         # Newton bindings (+ patches)
models/                                     # pretrained paper checkpoints
scripts/                                    # SLURM train + test scripts
  soccer_test.sh                            # local interactive demo viewer
  download_paper_models.bash                # re-fetch models/ from the cluster
```

## Citation / references

If you build on this, please cite:

- AMP — Peng et al., 2021, [arXiv:2104.02180](https://arxiv.org/abs/2104.02180)
- DribbleMaster — Wang et al., 2025, [arXiv:2505.12679](https://arxiv.org/abs/2505.12679)
- MimicKit — Peng, 2025, https://github.com/xbpeng/MimicKit
