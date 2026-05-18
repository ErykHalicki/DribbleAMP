import pandas as pd
import matplotlib.pyplot as plt
import sys

log_file = sys.argv[1] if len(sys.argv) > 1 else "log.txt"
df = pd.read_csv(log_file, sep=r'\s+')

fig, axes = plt.subplots(3, 3, figsize=(9, 6))
fig.suptitle("Training Log")

plots = [
    ("Train_Return",     "Train Return"),
    ("Test_Return",      "Test Return"),
    ("Critic_Loss",      "Critic Loss"),
    ("Actor_Loss",       "Actor Loss"),
    ("Disc_Loss",        "Disc Loss"),
    ("Disc_Agent_Acc",   "Disc Agent Acc"),
    ("Disc_Demo_Acc",    "Disc Demo Acc"),
    ("Disc_Reward_Mean", "Disc Reward Mean"),
    ("Train_Episode_Length", "Episode Length"),
]

for ax, (col, title) in zip(axes.flat, plots):
    ax.plot(df["Iteration"], df[col])
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_log.png", dpi=150)
print("Saved training_log.png")
plt.show()
