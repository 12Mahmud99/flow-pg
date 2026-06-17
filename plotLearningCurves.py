#!/usr/bin/env python
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_csv(path):
    """Load a single progress.csv, renaming columns if needed."""
    df = pd.read_csv(path)
    # Rename common step column names to 'step' for consistency
    if 'time/total_timesteps' in df.columns:
        df.rename(columns={'time/total_timesteps': 'step'}, inplace=True)
    elif 'step' not in df.columns:
        # Try to find any column that looks like step
        step_cols = [c for c in df.columns if 'step' in c.lower() or 'timestep' in c.lower()]
        if step_cols:
            df.rename(columns={step_cols[0]: 'step'}, inplace=True)
        else:
            raise ValueError("No step column found in CSV.")
    # Check for reward column
    if 'rollout/ep_rew_mean' not in df.columns:
        raise ValueError("CSV must contain 'rollout/ep_rew_mean' column.")
    return df

def load_directory(base_dir):
    """Find all progress.csv files recursively and return list of DataFrames."""
    csv_files = glob.glob(os.path.join(base_dir, "**", "progress.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No progress.csv found in {base_dir}")
    dfs = [load_csv(f) for f in csv_files]
    return dfs

def smooth_curve(values, window=10):
    """Apply a moving average smoothing."""
    if window <= 1:
        return values
    return pd.Series(values).rolling(window, min_periods=1).mean().values

def plot_data(data, label=None, output=None, smooth=0, xlabel='Training Steps', ylabel='Average Episode Reward'):
    """
    Plot one or more DataFrames.
    If multiple DataFrames are given, average them with standard deviation shading.
    If a single DataFrame is given, plot the mean reward curve.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if isinstance(data, list):
        # Multiple runs: align steps, average, compute std
        # Assume all have the same step range; we'll use the first run's steps
        steps = data[0]['step'].values
        rewards = []
        for df in data:
            # Align by merging on step (if they differ, interpolate)
            merged = pd.merge(pd.DataFrame({'step': steps}), df[['step', 'rollout/ep_rew_mean']], on='step', how='left')
            merged['rollout/ep_rew_mean'] = merged['rollout/ep_rew_mean'].interpolate(method='linear')
            rewards.append(merged['rollout/ep_rew_mean'].values)
        rewards = np.array(rewards)
        mean = np.mean(rewards, axis=0)
        std = np.std(rewards, axis=0)
        if smooth > 0:
            mean = smooth_curve(mean, smooth)
            std = smooth_curve(std, smooth)
        ax.plot(steps, mean, label=label)
        ax.fill_between(steps, mean-std, mean+std, alpha=0.3)
    else:
        # Single DataFrame
        df = data
        steps = df['step'].values
        reward = df['rollout/ep_rew_mean'].values
        if smooth > 0:
            reward = smooth_curve(reward, smooth)
        ax.plot(steps, reward, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from Flow-PG progress.csv")
    parser.add_argument('--logdir', type=str, required=True,
                        help='Path to directory containing progress.csv files (or a single progress.csv)')
    parser.add_argument('--label', type=str, default='DDPGPro',
                        help='Label for the curve')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (e.g., plot.png). If not specified, shows interactively')
    parser.add_argument('--smooth', type=int, default=0,
                        help='Moving average window for smoothing (0 = no smoothing)')
    args = parser.parse_args()

    path = args.logdir
    label = args.label

    if os.path.isfile(path) and path.endswith('.csv'):
        df = load_csv(path)
        plot_data(df, label=label, output=args.output, smooth=args.smooth)
    elif os.path.isdir(path):
        dfs = load_directory(path)
        if len(dfs) == 1:
            plot_data(dfs[0], label=label, output=args.output, smooth=args.smooth)
        else:
            plot_data(dfs, label=label, output=args.output, smooth=args.smooth)
    else:
        print("Error: path must be a .csv file or a directory containing progress.csv files.")

if __name__ == "__main__":
    main()