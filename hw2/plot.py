"""Generate training and eval plots from CSV logs.

Usage:
    python plot.py <logdir>
    python plot.py Logdir/run_213704_use_wandb=false
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_eval(df, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Eval Metrics (PPO + BC Pretraining, hammer-v3)', fontsize=14, fontweight='bold')

    # Success Rate
    ax = axes[0, 0]
    ax.plot(df['frame'], df['episode_success'], color='steelblue', linewidth=1, alpha=0.4)
    ax.plot(df['frame'], df['episode_success'].rolling(10, min_periods=1).mean(),
            color='steelblue', linewidth=2, label='10-eval avg')
    last20 = df['episode_success'].iloc[-20:].mean()
    ax.axhline(y=last20, color='red', linestyle='--', alpha=0.5, label=f'last-20 avg: {last20:.2f}')
    ax.set_title('Eval Success Rate')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Episode Reward
    ax = axes[0, 1]
    ax.plot(df['frame'], df['episode_reward'], color='darkorange', linewidth=1, alpha=0.4)
    ax.plot(df['frame'], df['episode_reward'].rolling(10, min_periods=1).mean(),
            color='darkorange', linewidth=2, label='10-eval avg')
    ax.set_title('Eval Episode Reward')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Success Rate Distribution
    ax = axes[1, 0]
    ax.hist(df['episode_success'], bins=20, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(df['episode_success'].mean(), color='red', linestyle='--',
               label=f'mean: {df["episode_success"].mean():.2f}')
    ax.set_title('Success Rate Distribution')
    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reward vs Success scatter
    ax = axes[1, 1]
    sc = ax.scatter(df['episode_success'], df['episode_reward'],
                    c=df['frame'], cmap='viridis', s=15, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Frame')
    ax.set_title('Reward vs Success Rate (colored by frame)')
    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / 'eval_plots.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_train(df, save_dir, eval_df=None):
    df['reward_smooth'] = df['episode_reward'].rolling(50, min_periods=1).mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Eval Success Rate (top-left)
    ax = axes[0, 0]
    if eval_df is not None and 'episode_success' in eval_df.columns:
        ax.plot(eval_df['frame'], eval_df['episode_success'], 'o-', color='steelblue',
                markersize=2, linewidth=0.8)
        ax.set_ylim(0, 1)
    ax.set_title('Eval Success Rate')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Success Rate')

    # Eval Episode Reward (top-right)
    ax = axes[0, 1]
    if eval_df is not None and 'episode_reward' in eval_df.columns:
        ax.plot(eval_df['frame'], eval_df['episode_reward'], 'o-', color='darkorange',
                markersize=2, linewidth=0.8)
    ax.set_title('Eval Episode Reward')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Reward')

    # Train Episode Reward (bottom-left)
    ax = axes[1, 0]
    ax.plot(df['frame'], df['episode_reward'], alpha=0.25, color='seagreen', linewidth=0.7)
    ax.plot(df['frame'], df['reward_smooth'], color='seagreen', linewidth=2, label='50-ep avg')
    ax.set_title('Train Episode Reward')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Reward')
    ax.legend()

    # FPS (bottom-right)
    ax = axes[1, 1]
    ax.fill_between(df['frame'], 0, df['fps'].clip(upper=2500), color='slateblue', alpha=0.5)
    ax.plot(df['frame'], df['fps'].clip(upper=2500), color='slateblue', linewidth=0.5, alpha=0.7)
    ax.set_title('Training Speed (FPS)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('FPS')

    plt.tight_layout()
    out = save_dir / 'train_plots.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def main():
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <logdir>')
        sys.exit(1)

    logdir = Path(sys.argv[1])
    eval_csv = logdir / 'eval.csv'
    train_csv = logdir / 'train.csv'

    eval_df = None
    if eval_csv.exists():
        eval_df = pd.read_csv(eval_csv)
        plot_eval(eval_df, logdir)
    else:
        print(f'No eval.csv found in {logdir}')

    if train_csv.exists():
        plot_train(pd.read_csv(train_csv), logdir, eval_df=eval_df)
    else:
        print(f'No train.csv found in {logdir}')


if __name__ == '__main__':
    main()
