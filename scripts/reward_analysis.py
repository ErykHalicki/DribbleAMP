import math


PHASE1 = {
    "name": "Phase 1",
    "ball_speed_w": 0.8,
    "ball_dir_w": 0.0,
    "ball_dist_w": 0.2,
    "ball_speed_scale": 0.5,
    "ball_dist_scale": 0.02,
}

PHASE2 = {
    "name": "Phase 2",
    "ball_speed_w": 0.3,
    "ball_dir_w": 0.6,
    "ball_dist_w": 0.1,
    "ball_speed_scale": 0.5,
    "ball_dist_scale": 0.02,
}


def compute(dist, ball_speed, dir_alignment, tar_speed, cfg):
    speed_err = (tar_speed - ball_speed) ** 2
    r_speed = math.exp(-cfg["ball_speed_scale"] * speed_err)

    r_dir = max(0.0, dir_alignment) if ball_speed > 1e-6 else 0.0

    r_dist = math.exp(-cfg["ball_dist_scale"] * dist**2)

    total = (cfg["ball_speed_w"] * r_speed
             + cfg["ball_dir_w"] * r_dir
             + cfg["ball_dist_w"] * r_dist)
    return r_speed, r_dir, r_dist, total


SCENARIOS = [
    # (label, dist_m, ball_speed_mps, dir_alignment, tar_speed_mps)
    ("Ball far, stationary, agent far",            10.0, 0.0,  0.0, 2.0),
    ("Ball close, stationary, agent close",         1.0, 0.0,  0.0, 2.0),
    ("Touching ball, stationary",                   0.0, 0.0,  0.0, 2.0),
    ("Sitting on ball, no movement",                0.0, 0.0,  0.0, 2.0),
    ("Ball at 3m, agent there, slow ball any dir",  3.0, 1.0,  0.0, 2.0),
    ("Kicked ball moving fast, right direction",    2.0, 2.0,  1.0, 2.0),
    ("Kicked ball moving fast, wrong direction",    2.0, 2.0, -1.0, 2.0),
    ("Kicked ball moving fast, 45 deg off",         2.0, 2.0,  0.707, 2.0),
    ("Ball moving correct dir but too slow",        2.0, 0.5,  1.0, 2.0),
    ("Ball moving correct dir but too fast",        2.0, 4.0,  1.0, 2.0),
    ("Perfect: close, right speed, right dir",      0.5, 2.0,  1.0, 2.0),
    ("Target speed 0 (stop ball), ball at rest",    0.5, 0.0,  0.0, 0.0),
    ("Target speed 0, ball still moving",           0.5, 2.0,  1.0, 0.0),
    ("Far away, ball moving right way at target",   8.0, 2.0,  1.0, 2.0),
]


def format_row(label, dist, speed, align, tar_speed, cfg):
    r_speed, r_dir, r_dist, total = compute(dist, speed, align, tar_speed, cfg)
    return (f"  speed={r_speed:.3f} (w={cfg['ball_speed_w']:.2f}) "
            f" dir={r_dir:.3f} (w={cfg['ball_dir_w']:.2f}) "
            f" dist={r_dist:.3f} (w={cfg['ball_dist_w']:.2f}) "
            f"-> total={total:.3f}")


def main():
    for cfg in (PHASE1, PHASE2):
        print(f"\n=== {cfg['name']} ===")
        print(f"  weights: speed={cfg['ball_speed_w']} dir={cfg['ball_dir_w']} dist={cfg['ball_dist_w']}")
        print(f"  scales:  speed={cfg['ball_speed_scale']} dist={cfg['ball_dist_scale']}")
        print()
        for label, dist, speed, align, tar_speed in SCENARIOS:
            print(f"{label}")
            print(f"  inputs: dist={dist}m  ball_speed={speed}m/s  dir_align={align}  tar_speed={tar_speed}m/s")
            print(format_row(label, dist, speed, align, tar_speed, cfg))
            print()


if __name__ == "__main__":
    main()
