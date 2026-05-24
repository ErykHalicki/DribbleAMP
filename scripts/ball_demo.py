"""Minimal Newton scene to inspect ball physics (rolling friction etc.).

Loads data/assets/objects/football.xml, drops the ball with an initial horizontal
velocity, and renders with newton.viewer.ViewerGL. Press Esc/close to exit.

Run from the project root:
    .venv/bin/python scripts/ball_demo.py

Or override the initial speed:
    BALL_SPEED=8.0 .venv/bin/python scripts/ball_demo.py
"""
import os

import numpy as np
import warp as wp
import newton


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_PATH = os.path.join(PROJECT_ROOT, "MimicKit", "data", "assets", "objects", "football.xml")
BALL_SPEED = float(os.environ.get("BALL_SPEED", "5.0"))
SIM_FREQ = 240
CONTROL_FREQ = 30
SIM_STEPS_PER_FRAME = SIM_FREQ // CONTROL_FREQ
SIM_DT = 1.0 / SIM_FREQ


def main():
    wp.init()
    device = wp.get_device()

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = 1.0

    ground_cfg = builder.ShapeConfig(mu=1.0, restitution=0)
    builder.add_ground_plane(cfg=ground_cfg)

    builder.add_mjcf(
        ASSET_PATH,
        floating=True,
        ignore_inertial_definitions=False,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        convert_3d_hinge_to_ball_joints=True,
    )

    model = builder.finalize(device=device, requires_grad=False)
    solver = newton.solvers.SolverMuJoCo(
        model, solver="newton",
        njmax=64, nconmax=64, impratio=10, iterations=50, ls_iterations=20,
    )

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.collide(state0)

    # Free joint q layout: [x, y, z, qw, qx, qy, qz], qd layout: [vx, vy, vz, wx, wy, wz].
    q = state0.joint_q.numpy().copy()
    qd = state0.joint_qd.numpy().copy()
    q[0:3] = [0.0, 0.0, 0.5]
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    qd[0:3] = [BALL_SPEED, 0.0, 0.0]
    qd[3:6] = [0.0, 0.0, 0.0]
    state0.joint_q.assign(q)
    state0.joint_qd.assign(qd)

    viewer = newton.viewer.ViewerGL(headless=False)
    viewer.set_model(model)

    sim_time = 0.0
    frame = 0
    print(f"Ball initial speed: {BALL_SPEED} m/s in +x. Watch how long it rolls.")
    try:
        while viewer.is_running():
            for _ in range(SIM_STEPS_PER_FRAME):
                contacts = model.collide(state0)
                solver.step(state0, state1, control, contacts, SIM_DT)
                state0, state1 = state1, state0
                sim_time += SIM_DT
            viewer.begin_frame(sim_time)
            viewer.log_state(state0)
            viewer.end_frame()

            if frame % CONTROL_FREQ == 0:
                v = state0.joint_qd.numpy()[0:3]
                speed = float(np.linalg.norm(v))
                print(f"  t={sim_time:5.2f}s  speed={speed:.3f} m/s")
            frame += 1
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
