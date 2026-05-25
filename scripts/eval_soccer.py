"""Quantitative evaluation for a trained soccer policy.

Runs the policy for a fixed number of steps and logs, at every step:
    - target velocity vector       (tar_speed * tar_dir, shape [N, 2])
    - actual ball velocity vector  (proj_vel[:, 0, 0:2])
    - char-to-ball distance        (||ball_xy - root_xy||)

Then reports aggregate metrics:
    - mean / std actual speed, target speed
    - mean speed error |target - actual|
    - mean direction alignment cos(target_dir, actual_dir)   (only when ball moving)
    - mean angle error in degrees
    - mean / median / p90 char-ball distance

Usage:
    .venv/bin/python scripts/eval_soccer.py \
        --model_file output/model.pt \
        --num_envs 16 \
        [--steps_per_episode 900 --num_episodes 20] \
        [--total_steps 5000]   # hard loop cap, overrides the multiply-out
        [--csv output/eval.csv]

Mirrors run.py's setup so the same env / agent / engine configs apply.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIMICKIT_ROOT = os.path.join(REPO_ROOT, "MimicKit")
sys.path.insert(0, os.path.join(MIMICKIT_ROOT, "mimickit"))

os.chdir(MIMICKIT_ROOT)

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.mp_util as mp_util


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_file", required=True)
    p.add_argument("--env_config", default="data/envs/amp_soccer_humanoid_env_phase1.yaml")
    p.add_argument("--engine_config", default="data/engines/newton_engine.yaml")
    p.add_argument("--agent_config", default="data/agents/amp_task_humanoid_agent.yaml")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--steps_per_episode", type=int, default=900,
                   help="Steps per episode (controls warmup re-trigger). 900 @ 30Hz = 30s.")
    p.add_argument("--num_episodes", type=int, default=20,
                   help="Number of episodes. Ignored if --total_steps is set.")
    p.add_argument("--total_steps", type=int, default=0,
                   help="If >0, overrides steps_per_episode * num_episodes as the hard loop cap.")
    p.add_argument("--episode_length", type=float, default=30.0,
                   help="Seconds. Overrides env config episode duration (controls env-side resets).")
    p.add_argument("--warmup_steps", type=int, default=30,
                   help="Steps to skip after each reset before logging (lets ball settle).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--csv", default="", help="Optional path to dump per-step records.")
    p.add_argument("--ball_spawn_dist", type=float, default=0.4,
                   help="Eval override: spawn ball this far from agent root, in target direction.")
    return p.parse_args()


def install_deterministic_spawn(env, dist):
    """Replace ball spawn with a fixed offset in the target direction.

    Original `_launch_projectiles` spawns at a random angle/distance around the
    character. For eval we want the ball always reachable on the first kick, so
    we place it `dist` meters from the root in the direction the target line
    points (between the agent and the goal direction).
    """
    char_id = env._get_char_id()

    def deterministic_launch(env_ids, proj_ids):
        n = env_ids.shape[0]
        if (n == 0):
            return
        root_pos = env._engine.get_root_pos(char_id)[env_ids]
        tar_dir = env._tar_ball_dir[env_ids]

        spawn_pos = torch.zeros([n, 3], device=env._device, dtype=torch.float)
        spawn_pos[:, 0] = root_pos[:, 0] + dist * tar_dir[:, 0]
        spawn_pos[:, 1] = root_pos[:, 1] + dist * tar_dir[:, 1]
        spawn_pos[:, 2] = env._proj_radius

        spawn_vel = torch.zeros([n, 3], device=env._device, dtype=torch.float)
        spawn_rot = torch.zeros([n, 4], device=env._device, dtype=torch.float)
        spawn_rot[:, 3] = 1.0
        spawn_ang_vel = torch.zeros([n, 3], device=env._device, dtype=torch.float)

        for local_proj_id, proj_obj_id in enumerate(env._proj_ids):
            curr_mask = proj_ids == local_proj_id
            if (curr_mask.any().item()):
                curr_env_ids = env_ids[curr_mask]
                env._engine.set_root_pos(curr_env_ids, proj_obj_id, spawn_pos[curr_mask])
                env._engine.set_root_rot(curr_env_ids, proj_obj_id, spawn_rot[curr_mask])
                env._engine.set_root_vel(curr_env_ids, proj_obj_id, spawn_vel[curr_mask])
                env._engine.set_root_ang_vel(curr_env_ids, proj_obj_id, spawn_ang_vel[curr_mask])
                env._prev_proj_vel[curr_env_ids, local_proj_id] = spawn_vel[curr_mask]

        env._proj_trigger_times[env_ids, proj_ids] = float("inf")

    env._launch_projectiles = deterministic_launch


def collect_step(env):
    proj_pos, proj_vel = env._get_proj_states()
    char_id = env._get_char_id()
    root_pos = env._engine.get_root_pos(char_id)

    ball_xy = proj_pos[:, 0, 0:2]
    ball_vel_xy = proj_vel[:, 0, 0:2]
    root_xy = root_pos[:, 0:2]

    tar_dir = env._tar_ball_dir
    tar_speed = env._tar_ball_speed
    tar_vel = tar_speed.unsqueeze(-1) * tar_dir

    dist = torch.norm(ball_xy - root_xy, dim=-1)

    return {
        "ball_vel": ball_vel_xy.detach().cpu().numpy(),
        "tar_vel": tar_vel.detach().cpu().numpy(),
        "dist": dist.detach().cpu().numpy(),
    }


def summarize(records):
    ball_vel = np.concatenate([r["ball_vel"] for r in records], axis=0)
    tar_vel = np.concatenate([r["tar_vel"] for r in records], axis=0)
    dist = np.concatenate([r["dist"] for r in records], axis=0)

    ball_speed = np.linalg.norm(ball_vel, axis=-1)
    tar_speed = np.linalg.norm(tar_vel, axis=-1)

    speed_err = np.abs(tar_speed - ball_speed)

    eps = 1e-6
    moving = ball_speed > 0.05
    ball_dir = ball_vel / np.clip(ball_speed[:, None], eps, None)
    tar_dir = tar_vel / np.clip(tar_speed[:, None], eps, None)
    cos_align = np.sum(ball_dir * tar_dir, axis=-1)
    cos_align_moving = cos_align[moving]
    angle_err_deg = np.degrees(np.arccos(np.clip(cos_align_moving, -1.0, 1.0)))

    fmt = lambda x: f"{x: .4f}"
    print("=" * 60)
    print(f"samples: {ball_speed.shape[0]}  (moving: {moving.sum()})")
    print("-" * 60)
    print(f"target speed   mean={fmt(tar_speed.mean())}  std={fmt(tar_speed.std())}")
    print(f"actual speed   mean={fmt(ball_speed.mean())}  std={fmt(ball_speed.std())}")
    print(f"speed error    mean={fmt(speed_err.mean())}  median={fmt(np.median(speed_err))}")
    print("-" * 60)
    if (cos_align_moving.size > 0):
        print(f"dir cos align  mean={fmt(cos_align_moving.mean())}  std={fmt(cos_align_moving.std())}")
        print(f"angle err deg  mean={fmt(angle_err_deg.mean())}  median={fmt(np.median(angle_err_deg))}  p90={fmt(np.percentile(angle_err_deg, 90))}")
    else:
        print("dir cos align  (no moving samples)")
    print("-" * 60)
    print(f"char-ball dist mean={fmt(dist.mean())}  median={fmt(np.median(dist))}  p90={fmt(np.percentile(dist, 90))}")
    print("=" * 60)


def maybe_write_csv(path, records):
    if (not path):
        return
    rows = []
    for step, r in enumerate(records):
        N = r["ball_vel"].shape[0]
        for i in range(N):
            rows.append({
                "step": step,
                "env": i,
                "tar_vx": r["tar_vel"][i, 0], "tar_vy": r["tar_vel"][i, 1],
                "ball_vx": r["ball_vel"][i, 0], "ball_vy": r["ball_vel"][i, 1],
                "dist": r["dist"][i],
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    args = parse_args()

    mp_util.init(0, 1, args.device, master_port=None)

    env = env_builder.build_env(args.env_config, args.engine_config,
                                num_envs=args.num_envs, device=args.device,
                                visualize=args.visualize, record_video=False)
    agent = agent_builder.build_agent(args.agent_config, env, args.device)
    agent.load(args.model_file)

    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)

    env._episode_length = float(args.episode_length)
    env._tar_change_time_min = 1.0e9
    env._tar_change_time_max = 1.0e9

    install_deterministic_spawn(env, dist=args.ball_spawn_dist)

    obs, info = agent._reset_envs()
    if (hasattr(env, "_tar_change_times")):
        env._tar_change_times[:] = float("inf")

    records = []
    if (args.total_steps > 0):
        total_steps = args.total_steps
    else:
        total_steps = args.steps_per_episode * args.num_episodes
    print(f"running {total_steps} total steps "
          f"({args.steps_per_episode} per episode, {args.num_envs} envs)")
    with torch.no_grad():
        for step in range(total_steps):
            action, _ = agent._decide_action(obs, info)
            _, _, done, _ = agent._step_env(action)

            step_in_ep = step % args.steps_per_episode
            if (step_in_ep >= args.warmup_steps):
                records.append(collect_step(env))

            obs, info = agent._reset_done_envs(done)
            if (hasattr(env, "_tar_change_times")):
                env._tar_change_times[:] = float("inf")

            if (step % 100 == 0):
                print(f"step {step}/{total_steps}")

    summarize(records)
    maybe_write_csv(args.csv, records)


if __name__ == "__main__":
    main()
