# Ball Dribbling Environment — Implementation Plan

## Goal
Train a humanoid (via AMP) to dribble a ball toward a moving target on the ground, blending an AMP locomotion motion-prior with a task reward that drives the ball to the target.

## Anchor: what we are reusing
The `amp_location_humanoid_args.txt` pipeline is the closest existing recipe. It chains:

- `args/amp_location_humanoid_args.txt` — entry point
- `data/envs/amp_location_humanoid_env.yaml` — env config (`env_name: task_location`)
- `data/agents/amp_task_humanoid_agent.yaml` — AMP+task agent (no change needed)
- `mimickit/envs/task_location_env.py` — Python env class

We will build a parallel `task_dribble` slice that mirrors this structure, borrowing the **target-marker + reward** pattern from `task_location_env.py` and the **per-env free-floating rigid object + reset/launch lifecycle** from `task_dodgeball_env.py`.

## New files

| Layer | New file | Modeled on |
|---|---|---|
| Args | `args/amp_dribble_humanoid_args.txt` | `args/amp_location_humanoid_args.txt` |
| Env config | `data/envs/amp_dribble_humanoid_env.yaml` | `amp_location_humanoid_env.yaml` (+ ball params) |
| Asset | `data/assets/objects/soccer_ball.xml` | `data/assets/objects/dodgeball.xml` (lower density, smaller radius, friction/restitution tuned) |
| Env class | `mimickit/envs/task_dribble_env.py` | `task_location_env.py` + projectile lifecycle from `task_dodgeball_env.py` |
| Agent | reuse `data/agents/amp_task_humanoid_agent.yaml` | — |

One edit only: register `"task_dribble"` in `mimickit/envs/env_builder.py:38-49` next to the other `task_*` branches.

## Env class sketch (`TaskDribbleEnv(SMPEnv)`)

State, per env:
- `_ball_id` (rigid body, free joint, gravity on, like `dodgeball.xml`)
- `_marker_id` (visual-only flag, like `task_location_env._build_marker`)
- `_tar_pos [N,3]` — goal position on ground
- `_tar_change_times [N]` — same scheduling as location env
- (no `_prev_*` tensors needed — reward is positional only)

Lifecycle hooks to override (names match base class conventions seen in the two reference envs):
- `_build_env` — create marker + ball per env
- `_build_sim_tensors` — allocate `_tar_pos`, `_tar_change_times`, `_prev_*`
- `_reset_envs` → `_reset_task` → `_reset_tar` + `_reset_ball`
  - Place ball ~0.5–1.0 m in front of the character at reset
  - Sample target on a ring around the char (mirror `_reset_tar` in location env)
- `_update_misc` → `_update_task` — re-sample target on a 5–10 s schedule (copy from location env)
- `_compute_obs` — append task obs (see below)
- `_update_reward` — composite reward (see below)
- `_render_scene` — draw a line char→ball and ball→target to debug

## Observations (appended to AMP base obs)
All in the character's heading frame (`torch_util.calc_heading_quat_inv`), as `task_location_env` does:

1. `local_ball_pos_xy` (2)
2. `local_target_pos_xy` (2)
3. `local_ball_to_target_xy` (2)  — redundant but cheap and helps the policy

Total: 6 task obs, concatenated onto `super()._compute_obs(env_ids)`. Ball velocity is intentionally omitted from obs since it isn't part of the reward; add it back later if the policy needs it.

## Reward (task reward = ball-to-target distance, only)
Single term:
```
pos_err     = ||ball_xy - tar_xy||²
task_reward = exp(-pos_err_scale * pos_err)
```
`pos_err_scale` ~0.5 to start (matches `task_location` defaults).

No ball-velocity, char-near-ball, or foot-proximity terms.

The **AMP discriminator reward** is still mixed in by the agent automatically via `task_reward_weight` / `disc_reward_weight` in `amp_task_humanoid_agent.yaml` — that's structural to AMP, not a task reward, so it stays.

## Termination
Inherit fall/pose termination from base. Add an optional fail flag if `||root - ball|| > ball_lost_dist` (default 4 m) so the agent can't run away from a stuck ball. Set via `_done_buf[lost] = base_env.DoneFlags.FAIL.value`, same pattern as `task_dodgeball_env._update_done`.

## Asset
Start by copying `dodgeball.xml` to `soccer_ball.xml` and adjust:
- `size="0.11"` (FIFA size 5 ≈ 22 cm dia)
- `density` so mass ≈ 0.43 kg → density ≈ 77 for r=0.11
- Add `<geom ... friction="0.8 0.005 0.0001" solref="0.005 1" solimp="0.95 0.99 0.001"/>` so it rolls and bounces sensibly. Iterate after first runs.

If Isaac Gym is the engine, also produce `soccer_ball.usd` via the existing tooling that produced `dodgeball.usd` — check `tools/` for the conversion script.

## Motion data
Use `data/datasets/dataset_humanoid_locomotion.yaml` (already used by `amp_location_humanoid_env.yaml`). Walking/jogging clips give the discriminator the right style. No new motion capture needed for v1.

## Bring-up order (smallest-step first)
1. Asset only: copy `soccer_ball.xml`, load it inside `task_location_env` temporarily as a passive prop, run `--visualize true --num_envs 4` to confirm it spawns and falls correctly.
2. Wire the new env class as a clone of `task_location_env` that *ignores* the ball — train briefly to confirm the AMP+task pipeline still converges to "walk to marker".
3. Replace the location-based reward with the ball-target reward from §Reward. Train.
4. Tune ball physics + reward weights based on visual rollouts (`--visualize true`, small `--num_envs`).
5. Add the foot-proximity term and lost-ball termination.

## Known unknowns (resolve by reading code, not guessing)
- Whether `engine.create_obj` with `obj_type=rigid` on the Newton/Isaac-Lab backends supports per-env reset with the same API used in `task_dodgeball_env`. The dodgeball env asserts identical IDs across envs — verify this still holds for our backend before relying on it. (The TODO note "run amp using isaac lab" suggests backend portability is in flux.)
- Contact pair filtering: the ball must collide with feet but not be glued by the character's capsule contacts. Check `char_file` `humanoid.xml` contype/conaffinity.
- `_get_proj_contact_force` is a useful template if we later want a "kick" reward shaped on impulse rather than position.

## Out of scope for v1
- Multi-ball / opponents
- Ball spin observation
- Goal-shaped target (use the existing flat marker)
- Curriculum on target distance (can add later by scheduling `tar_dist_max`)
