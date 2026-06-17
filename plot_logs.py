import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (adjust filename if needed)
logs = pd.read_csv("outputs/train_rl/2026-06-16/23-44-25/progress.csv")

# Check what columns exist
print(logs.columns)

# Plot episode reward
plt.figure(figsize=(10, 6))
plt.plot(logs['step'], logs['ep_rew_mean'], label='Mean Reward')
plt.fill_between(
    logs['step'],
    logs['ep_rew_mean'] - logs['ep_rew_std'],
    logs['ep_rew_mean'] + logs['ep_rew_std'],
    alpha=0.3
)
plt.xlabel('Step')
plt.ylabel('Episode Reward')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show()