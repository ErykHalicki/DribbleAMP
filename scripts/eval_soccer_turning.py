"""Dribbling trajectory eval with multiple samples and canonical-frame norm.

For each of 5 conditions (straight, left 45/90, right 45/90), runs
`--num_samples` rollouts in parallel envs:
    - 10 s with target direction = +x (in the character's initial heading frame)
    - 10 s with target direction rotated by the condition's signed angle

All trajectories are transformed into a canonical frame where every env starts
at the origin with its character facing +x, so trajectories can be averaged
across rollouts.

Outputs (under `--out_dir`):
    - trajectories.png  mean trajectory per condition + std band, individual
                        rollouts shown lightly.
    - summary.txt       per-condition mean speed.
"""

import argparse
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITIONS = [
    ("straight",  0.0),
    ("left 45",  +45.0),
    ("left 90",  +90.0),
    ("right 45", -45.0),
    ("right 90", -90.0),
]
COLORS = ["black", "tab:blue", "tab:cyan", "tab:red", "tab:orange"]
TARGET_SPEED = 1.0
BALL_SPAWN_DIST = 0.4
HZ = 30


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_file", required=True)
    p.add_argument("--env_config", default="data/envs/amp_soccer_humanoid_env_phase2.yaml")
    p.add_argument("--engine_config", default="data/engines/newton_engine.yaml")
    p.add_argument("--agent_config", default="data/agents/amp_task_humanoid_agent.yaml")
    p.add_argument("--num_samples", type=int, default=5,
                   help="Rollouts per condition (envs = num_samples * 5).")
    p.add_argument("--seconds_per_phase", type=float, default=10.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--out_dir", default="output/eval_turning")
    return p.parse_args()


def install_deterministic_spawn(env, dist):
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


def yaw_from_quat(q):
    """Yaw (rotation about +z) from quaternion stored as (x, y, z, w)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_to_local_dirs(local_dirs, yaw):
    """Rotate local-frame unit directions by yaw to get world-frame directions.
    local_dirs: [N, 2], yaw: [N]."""
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.zeros_like(local_dirs)
    out[:, 0] = c * local_dirs[:, 0] - s * local_dirs[:, 1]
    out[:, 1] = s * local_dirs[:, 0] + c * local_dirs[:, 1]
    return out


def rotate_points(points_xy, yaw):
    """Rotate [T, N, 2] points by -yaw[N] about the origin (world->local).
    Equivalently apply R(-yaw)."""
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    x = points_xy[..., 0]
    y = points_xy[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def force_targets_world(env, world_dirs_xy, speed):
    n = world_dirs_xy.shape[0]
    env._tar_ball_dir[:n, 0] = torch.from_numpy(world_dirs_xy[:, 0].astype(np.float32)).to(env._device)
    env._tar_ball_dir[:n, 1] = torch.from_numpy(world_dirs_xy[:, 1].astype(np.float32)).to(env._device)
    env._tar_ball_speed[:n] = float(speed)
    if (hasattr(env, "_tar_change_times")):
        env._tar_change_times[:] = float("inf")


def main():
    args = parse_args()
    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    mp_util.init(0, 1, args.device, master_port=None)

    num_conditions = len(CONDITIONS)
    num_envs = num_conditions * args.num_samples
    env = env_builder.build_env(args.env_config, args.engine_config,
                                num_envs=num_envs, device=args.device,
                                visualize=args.visualize, record_video=False)
    agent = agent_builder.build_agent(args.agent_config, env, args.device)
    agent.load(args.model_file)
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)

    total_seconds = 2.0 * args.seconds_per_phase
    env._episode_length = total_seconds + 5.0
    env._tar_change_time_min = 1.0e9
    env._tar_change_time_max = 1.0e9

    install_deterministic_spawn(env, dist=BALL_SPAWN_DIST)

    cond_idx = np.repeat(np.arange(num_conditions), args.num_samples)  # [num_envs]
    cond_angle_deg = np.array([c[1] for c in CONDITIONS], dtype=np.float32)
    per_env_angle_rad = np.deg2rad(cond_angle_deg[cond_idx])

    char_id = env._get_char_id()

    init_dirs_local = np.tile(np.array([1.0, 0.0], dtype=np.float32)[None, :], (num_envs, 1))
    root_rot = env._engine.get_root_rot(char_id).detach().cpu().numpy()
    yaw = yaw_from_quat(root_rot)
    init_dirs_world = world_to_local_dirs(init_dirs_local, yaw)
    force_targets_world(env, init_dirs_world, TARGET_SPEED)

    obs, info = agent._reset_envs()

    root_rot = env._engine.get_root_rot(char_id).detach().cpu().numpy()
    yaw = yaw_from_quat(root_rot)
    init_dirs_world = world_to_local_dirs(init_dirs_local, yaw)
    force_targets_world(env, init_dirs_world, TARGET_SPEED)

    root_pos = env._engine.get_root_pos(char_id).detach().cpu().numpy()
    char_origin_xy = root_pos[:, 0:2].copy()

    phase2_dirs_local = np.stack([np.cos(per_env_angle_rad), np.sin(per_env_angle_rad)], axis=-1)
    phase2_dirs_world = world_to_local_dirs(phase2_dirs_local.astype(np.float32), yaw)

    steps_per_phase = int(round(args.seconds_per_phase * HZ))
    total_steps = 2 * steps_per_phase

    ball_traj_world = np.zeros((total_steps, num_envs, 2), dtype=np.float32)
    speeds_per_env = [[] for _ in range(num_envs)]

    with torch.no_grad():
        for step in range(total_steps):
            if (step == steps_per_phase):
                force_targets_world(env, phase2_dirs_world, TARGET_SPEED)

            action, _ = agent._decide_action(obs, info)
            _, _, done, _ = agent._step_env(action)

            proj_pos, proj_vel = env._get_proj_states()
            ball_xy = proj_pos[:, 0, 0:2].detach().cpu().numpy()
            ball_v = proj_vel[:, 0, 0:2].detach().cpu().numpy()
            ball_traj_world[step] = ball_xy

            for i in range(num_envs):
                speeds_per_env[i].append(float(np.linalg.norm(ball_v[i])))

            obs, info = agent._reset_done_envs(done)
            if (step < steps_per_phase):
                force_targets_world(env, init_dirs_world, TARGET_SPEED)
            else:
                force_targets_world(env, phase2_dirs_world, TARGET_SPEED)

            if (step % 50 == 0):
                print(f"step {step}/{total_steps}")

    rel_world = ball_traj_world - char_origin_xy[None, :, :]
    rel_local = rotate_points(rel_world, yaw)

    n_cond = len(CONDITIONS)
    fig, axes = plt.subplots(1, n_cond, figsize=(4.0 * n_cond, 4.5), squeeze=False)
    axes = axes[0]

    all_pts = rel_local.reshape(-1, 2)
    pad = 0.3
    xlim = (float(all_pts[:, 0].min()) - pad, float(all_pts[:, 0].max()) + pad)
    ylim = (float(all_pts[:, 1].min()) - pad, float(all_pts[:, 1].max()) + pad)

    for c_idx, ((label, _), color, ax) in enumerate(zip(CONDITIONS, COLORS, axes)):
        env_mask = cond_idx == c_idx
        trials = rel_local[:, env_mask, :]  # [T, S, 2]

        ax.set_title(label)
        ax.set_xlabel("x (m)")
        if (c_idx == 0):
            ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)

        mean_traj = trials.mean(axis=1)
        std_traj = trials.std(axis=1)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=color, linewidth=2.2, label="mean")
        ax.fill_between(mean_traj[:, 0],
                        mean_traj[:, 1] - std_traj[:, 1],
                        mean_traj[:, 1] + std_traj[:, 1],
                        color=color, alpha=0.18, label="±1σ")
        ax.scatter([mean_traj[steps_per_phase, 0]], [mean_traj[steps_per_phase, 1]],
                   color=color, marker="x", s=50, label="turn")
        ax.scatter([0], [0], color="k", marker="o", s=30)
        ax.scatter([BALL_SPAWN_DIST], [0], color="gray", marker="s", s=20)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Dribbling trajectories ({args.num_samples} rollouts/condition)\n"
                 f"{args.seconds_per_phase:.0f} s straight + {args.seconds_per_phase:.0f} s post-turn")
    fig.tight_layout()
    path = os.path.join(out_dir, "trajectories.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")

    summary_lines = ["condition       mean speed (m/s)"]
    for c_idx, (label, _) in enumerate(CONDITIONS):
        env_mask = cond_idx == c_idx
        speeds = np.concatenate([np.array(speeds_per_env[i]) for i in np.where(env_mask)[0]])
        summary_lines.append(f"  {label:<13} {speeds.mean():.3f}  +/- {speeds.std():.3f}")
    text = "\n".join(summary_lines)
    print(text)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
