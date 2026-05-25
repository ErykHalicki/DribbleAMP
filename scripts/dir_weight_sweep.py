import math

# Fixed scenario knobs:
TAR_SPEED = 2.0
BALL_DIST = 1.0
SPEED_SCALE = 0.2
DIST_SCALE = 0.1

# Reward composition: (speed_w, dir_w, dist_w) — always sums to 1.0 so totals are comparable.
WEIGHTS = [
    (0.7, 0.0, 0.3),  # phase 1 (current)
    (0.6, 0.1, 0.3),
    (0.5, 0.2, 0.3),
    (0.4, 0.3, 0.3),
    (0.3, 0.4, 0.3),
    (0.2, 0.5, 0.3),
    (0.0, 0.7, 0.3),
]

# Scenarios share the same speed (target-matching) but vary direction.
SCENARIOS = [
    ("ball at rest",                          0.0,  0.0),   # speed=0, alignment irrelevant
    ("perfect: target speed, target dir",     2.0,  1.0),
    ("right speed, 45° off",                  2.0,  0.707),
    ("right speed, 90° off",                  2.0,  0.0),
    ("right speed, backwards",                2.0, -1.0),
    ("half speed, target dir",                1.0,  1.0),
    ("twice speed, target dir",               4.0,  1.0),
]


def reward(speed_w, dir_w, dist_w, ball_speed, dir_align):
    speed_err = (TAR_SPEED - ball_speed) ** 2
    r_speed = math.exp(-SPEED_SCALE * speed_err)
    r_dir = max(0.0, dir_align) if ball_speed > 1e-6 else 0.0
    r_dist = math.exp(-DIST_SCALE * BALL_DIST ** 2)
    return speed_w * r_speed + dir_w * r_dir + dist_w * r_dist


header = ["scenario"] + [f"sp={s} dr={d}" for s, d, _ in WEIGHTS]
col_w = max(len(h) for h in header[1:])
print(f"{'scenario':<38} | " + " | ".join(f"{h:<{col_w}}" for h in header[1:]))
print("-" * (40 + (col_w + 3) * len(WEIGHTS)))

for label, ball_speed, dir_align in SCENARIOS:
    row = [f"{label:<38}"]
    for sw, dw, dstw in WEIGHTS:
        total = reward(sw, dw, dstw, ball_speed, dir_align)
        row.append(f"{total:<{col_w}.3f}")
    print(" | ".join(row))

print()
print(f"Fixed: tar_speed={TAR_SPEED} m/s, ball_dist={BALL_DIST}m, speed_scale={SPEED_SCALE}, dist_scale={DIST_SCALE}")
print("All weight triples sum to 1.0 so values are comparable.")
