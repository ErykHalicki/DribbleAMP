import math

SCALES = [0.1, 0.25, 0.5, 1.0, 2.0]
SPEED_ERRS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

print(f"{'speed_err (m/s)':<18} | " + " | ".join(f"scale={s:<6}" for s in SCALES))
print("-" * (18 + 11 * len(SCALES)))

for err in SPEED_ERRS:
    row = [f"{err:<18}"]
    for scale in SCALES:
        r = math.exp(-scale * err**2)
        row.append(f"{r:<12.3f}")
    print(" | ".join(row))

print()
print("Reward formula: exp(-scale * (tar_speed - actual_speed)^2)")
print("err = |tar_speed - actual_speed| in m/s")
