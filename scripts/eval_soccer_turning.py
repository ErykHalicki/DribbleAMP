"""Velocity Tracking Accuracy evaluation (dribbling with sharp turns).

Mirrors the protocol from the reference paper:
    - Ball starts at the origin, commanded to move along +x at ~1 m/s.
    - When the ball enters a 0.4 m radius circle centered at (1.5, 0) relative
      to its spawn point, the target direction switches to a 45 or 90 degree
      turn (left or right).
    - 5 rollouts per (turn direction, turn angle), totalling 20 trials, with
      +/-0.1 m/s random target-speed perturbation.

Outputs:
    - Per-condition mean trajectory plot with shaded variance band.
    - Direction tracking error (target angle vs angle between fitted pre/post
      turn lines).
    - Speed tracking error (mean ball speed vs commanded speed).

Usage:
    .venv/bin/python scripts/eval_soccer_turning.py \\
        --model_file output/model.pt \\
        [--num_trials_per_cond 5] \\
        [--out_dir output/eval_turning]
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
    ("left",  45),
    ("left",  90),
    ("right", 45),
    ("right", 90),
]
TARGET_SPEED = 1.0
SPEED_PERTURB = 0.1
TURN_TRIGGER_CENTER = np.array([1.5, 0.0], dtype=np.float32)
TURN_TRIGGER_RADIUS = 0.4
INITIAL_DIR = np.array([1.0, 0.0], dtype=np.float32)
BALL_SPAWN_DIST = 0.4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_file", required=True)
    p.add_argument("--env_config", default="data/envs/amp_soccer_humanoid_env_phase1.yaml")
    p.add_argument("--engine_config", default="data/engines/newton_engine.yaml")
    p.add_argument("--agent_config", default="data/agents/amp_task_humanoid_agent.yaml")
    p.add_argument("--num_trials_per_cond", type=int, default=5)
    p.add_argument("--episode_steps", type=int, default=900,
                   help="Hard cap per rollout in env steps (30 Hz).")
    p.add_argument("--warmup_steps", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--out_dir", default="output/eval_turning")
    p.add_argument("--seed", type=int, default=0)
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


def rotate_2d(v, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)


def fit_line_angle(points):
    """Return angle (rad) of best-fit line through 2D points, oriented along
    the direction of travel (first-to-last)."""
    if (points.shape[0] < 2):
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    travel = points[-1] - points[0]
    if (np.dot(direction, travel) < 0):
        direction = -direction
    return float(np.arctan2(direction[1], direction[0]))


def run_condition(env, agent, turn_side, turn_angle_deg, num_trials, args, rng):
    """Run num_trials parallel rollouts for one (side, angle) condition."""
    num_envs = env.get_num_envs()
    assert num_envs >= num_trials, f"need num_envs >= {num_trials}"

    sign = +1.0 if turn_side == "left" else -1.0
    turn_angle_rad = sign * np.deg2rad(turn_angle_deg)

    speed_perturb = rng.uniform(-SPEED_PERTURB, SPEED_PERTURB, size=num_trials).astype(np.float32)
    tar_speed = TARGET_SPEED + speed_perturb

    env._tar_ball_dir[:num_trials, 0] = float(INITIAL_DIR[0])
    env._tar_ball_dir[:num_trials, 1] = float(INITIAL_DIR[1])
    env._tar_ball_speed[:num_trials] = torch.from_numpy(tar_speed).to(env._device)

    obs, info = agent._reset_envs()
    if (hasattr(env, "_tar_change_times")):
        env._tar_change_times[:] = float("inf")

    env._tar_ball_dir[:num_trials, 0] = float(INITIAL_DIR[0])
    env._tar_ball_dir[:num_trials, 1] = float(INITIAL_DIR[1])
    env._tar_ball_speed[:num_trials] = torch.from_numpy(tar_speed).to(env._device)

    proj_pos, _ = env._get_proj_states()
    spawn_xy = proj_pos[:num_trials, 0, 0:2].detach().cpu().numpy().copy()

    triggered = np.zeros(num_trials, dtype=bool)
    trigger_step = np.full(num_trials, -1, dtype=np.int64)
    traj = [[] for _ in range(num_trials)]
    speed_samples = [[] for _ in range(num_trials)]

    with torch.no_grad():
        for step in range(args.episode_steps):
            action, _ = agent._decide_action(obs, info)
            _, _, done, _ = agent._step_env(action)

            proj_pos, proj_vel = env._get_proj_states()
            ball_xy = proj_pos[:num_trials, 0, 0:2].detach().cpu().numpy()
            ball_v = proj_vel[:num_trials, 0, 0:2].detach().cpu().numpy()

            rel_xy = ball_xy - spawn_xy
            in_trigger = np.linalg.norm(rel_xy - TURN_TRIGGER_CENTER, axis=-1) < TURN_TRIGGER_RADIUS
            new_trig_mask = in_trigger & (~triggered)
            if (new_trig_mask.any()):
                for i in np.where(new_trig_mask)[0]:
                    new_dir = rotate_2d(INITIAL_DIR, turn_angle_rad)
                    env._tar_ball_dir[i, 0] = float(new_dir[0])
                    env._tar_ball_dir[i, 1] = float(new_dir[1])
                    trigger_step[i] = step
                triggered |= new_trig_mask

            if (step >= args.warmup_steps):
                for i in range(num_trials):
                    traj[i].append(rel_xy[i].copy())
                    speed_samples[i].append(float(np.linalg.norm(ball_v[i])))

            obs, info = agent._reset_done_envs(done)
            if (hasattr(env, "_tar_change_times")):
                env._tar_change_times[:] = float("inf")

    trials = []
    for i in range(num_trials):
        points = np.array(traj[i], dtype=np.float32)
        ts = trigger_step[i]
        if (ts < 0 or points.shape[0] < 10):
            pre_pts = points[: max(1, points.shape[0] // 2)]
            post_pts = points[max(1, points.shape[0] // 2):]
        else:
            log_trig = ts - args.warmup_steps
            pad = 5
            pre_pts = points[: max(2, log_trig - pad)]
            post_pts = points[log_trig + pad:]
            if (post_pts.shape[0] < 5):
                post_pts = points[log_trig:]

        pre_angle = fit_line_angle(pre_pts)
        post_angle = fit_line_angle(post_pts)
        measured_turn_deg = np.degrees((post_angle - pre_angle + np.pi) % (2 * np.pi) - np.pi)

        trials.append({
            "traj": points,
            "trigger_step": ts,
            "pre_angle": pre_angle,
            "post_angle": post_angle,
            "measured_turn_deg": float(measured_turn_deg),
            "tar_speed": float(tar_speed[i]),
            "mean_speed": float(np.mean(speed_samples[i])) if speed_samples[i] else 0.0,
            "triggered": bool(triggered[i]),
        })

    return {
        "side": turn_side,
        "angle_deg": turn_angle_deg,
        "signed_target_deg": sign * turn_angle_deg,
        "trials": trials,
    }


def resample_trajectory(points, n_samples=200):
    if (points.shape[0] < 2):
        return np.zeros((n_samples, 2), dtype=np.float32)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if (total < 1e-6):
        return np.repeat(points[0:1], n_samples, axis=0)
    u = np.linspace(0.0, total, n_samples)
    out = np.zeros((n_samples, 2), dtype=np.float32)
    for d in range(2):
        out[:, d] = np.interp(u, arc, points[:, d])
    return out


def plot_results(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sides = ["left", "right"]
    angle_colors = {45: "tab:blue", 90: "tab:orange"}

    for ax, side in zip(axes, sides):
        ax.set_title(f"{side.capitalize()}-turn dribbling trajectory")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        circ = plt.Circle(TURN_TRIGGER_CENTER, TURN_TRIGGER_RADIUS, color="gray",
                          fill=False, linestyle="--", linewidth=1.0, label="turn trigger")
        ax.add_patch(circ)

        for angle in (45, 90):
            cond = next(r for r in results if r["side"] == side and r["angle_deg"] == angle)
            resampled = np.stack([resample_trajectory(t["traj"]) for t in cond["trials"]], axis=0)
            mean_traj = resampled.mean(axis=0)
            std_traj = resampled.std(axis=0)

            color = angle_colors[angle]
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=color, linewidth=2.0,
                    label=f"{angle}°")
            ax.fill_between(mean_traj[:, 0],
                            mean_traj[:, 1] - std_traj[:, 1],
                            mean_traj[:, 1] + std_traj[:, 1],
                            color=color, alpha=0.2)
        ax.legend(loc="best")

    fig.tight_layout()
    path = os.path.join(out_dir, "trajectories.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def summarize(results, out_dir):
    lines = []
    lines.append("=" * 70)
    lines.append(f"{'condition':<18}{'tar deg':>10}{'meas deg':>12}"
                 f"{'dir err %':>12}{'mean spd':>12}{'spd err %':>12}")
    lines.append("-" * 70)
    for r in results:
        target = r["signed_target_deg"]
        measured = np.array([t["measured_turn_deg"] for t in r["trials"]])
        mean_meas = float(np.mean(np.abs(measured)))
        dir_rel_err = abs(abs(target) - mean_meas) / abs(target) * 100.0

        speeds = np.array([t["mean_speed"] for t in r["trials"]])
        targets = np.array([t["tar_speed"] for t in r["trials"]])
        mean_spd = float(np.mean(speeds))
        spd_rel_err = float(np.mean(np.abs(targets - speeds) / targets)) * 100.0

        lines.append(f"{r['side']}-{r['angle_deg']:<12}{target:>10.1f}{mean_meas:>12.2f}"
                     f"{dir_rel_err:>12.2f}{mean_spd:>12.3f}{spd_rel_err:>12.2f}")

        for ti, t in enumerate(r["trials"]):
            lines.append(f"    trial {ti}: triggered={t['triggered']} "
                         f"meas={t['measured_turn_deg']:.2f}  "
                         f"tar_spd={t['tar_speed']:.3f}  mean_spd={t['mean_speed']:.3f}")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")


def main():
    args = parse_args()
    os.makedirs(os.path.join(REPO_ROOT, args.out_dir), exist_ok=True)
    out_dir_abs = os.path.join(REPO_ROOT, args.out_dir)
    rng = np.random.default_rng(args.seed)

    mp_util.init(0, 1, args.device, master_port=None)

    num_envs = args.num_trials_per_cond
    env = env_builder.build_env(args.env_config, args.engine_config,
                                num_envs=num_envs, device=args.device,
                                visualize=args.visualize, record_video=False)
    agent = agent_builder.build_agent(args.agent_config, env, args.device)
    agent.load(args.model_file)
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)

    env._episode_length = float(args.episode_steps) / 30.0 + 5.0
    env._tar_change_time_min = 1.0e9
    env._tar_change_time_max = 1.0e9

    install_deterministic_spawn(env, dist=BALL_SPAWN_DIST)

    results = []
    for side, angle in CONDITIONS:
        print(f"\n=== running {side}-turn {angle}° ({args.num_trials_per_cond} trials) ===")
        cond_result = run_condition(env, agent, side, angle,
                                    args.num_trials_per_cond, args, rng)
        results.append(cond_result)

    summarize(results, out_dir_abs)
    plot_results(results, out_dir_abs)


if __name__ == "__main__":
    main()
