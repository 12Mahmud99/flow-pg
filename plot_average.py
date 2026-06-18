import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import argparse

def load_and_average(directory, label):
    """Find all progress.csv in directory (recursively) and average them."""
    files = glob.glob(os.path.join(directory, "**", "progress.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No progress.csv found in {directory}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if 'time/total_timesteps' in df.columns:
            df.rename(columns={'time/total_timesteps': 'step'}, inplace=True)
        if 'step' not in df.columns:
            raise ValueError(f"No step column in {f}")
        dfs.append(df.set_index('step')['rollout/ep_rew_mean'])
    combined = pd.concat(dfs, axis=1)
    mean = combined.mean(axis=1)
    std = combined.std(axis=1)
    return mean, std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Directory containing run subfolders')
    parser.add_argument('--label', type=str, default='DDPGPro', help='Label for the curve')
    parser.add_argument('--output', type=str, default='plot.png', help='Output file name')
    args = parser.parse_args()
    
    mean, std = load_and_average(args.dir, args.label)
    
    plt.figure(figsize=(10,6))
    plt.plot(mean.index, mean.values, label=args.label)
    plt.fill_between(mean.index, mean-std, mean+std, alpha=0.3)
    plt.xlabel('Training Steps')
    plt.ylabel('Average Episode Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.output, dpi=150)
    plt.show()

if __name__ == "__main__":
    main()