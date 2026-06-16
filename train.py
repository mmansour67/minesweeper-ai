"""
train.py
========
Trains the DQN agent by letting it play many games of Minesweeper. As training
proceeds it should win an increasing fraction of games -- that improving win
rate is the whole point of the project.

Run it like:
    python train.py                       # default 6x6 board, 6 mines
    python train.py --rows 9 --cols 9 --mines 10 --episodes 50000

What it produces:
    * console output every N games showing the recent win rate
    * minesweeper_dqn.pt   : the saved trained model
    * training_log.csv     : episode, win rate, average reward, epsilon
"""

from __future__ import annotations

import argparse
import csv
from collections import deque

import numpy as np

from minesweeper import Minesweeper
from dqn_agent import DQNAgent, encode_state


def main():
    parser = argparse.ArgumentParser(description="Train a DQN to play Minesweeper.")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--mines", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=20_000,
                        help="number of games to play during training")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="safety cap on moves per game")
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.01)
    parser.add_argument("--eps-decay", type=int, default=15_000,
                        help="games over which exploration fades to eps-end")
    parser.add_argument("--batch", type=int, default=256,
                        help="replay batch size (smaller = faster per step on CPU)")
    parser.add_argument("--report-every", type=int, default=500)
    parser.add_argument("--model-out", type=str, default="minesweeper_dqn.pt")
    parser.add_argument("--log-out", type=str, default="training_log.csv")
    args = parser.parse_args()

    env = Minesweeper(rows=args.rows, cols=args.cols, n_mines=args.mines)
    agent = DQNAgent(rows=args.rows, cols=args.cols, batch_size=args.batch)

    # Sliding windows for reporting recent performance.
    recent_wins = deque(maxlen=args.report_every)
    recent_rewards = deque(maxlen=args.report_every)

    log_rows = []
    print(f"Training on a {args.rows}x{args.cols} board with {args.mines} mines.")
    print(f"Device: {agent.device}\n")

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        # Linearly decay exploration from eps-start down to eps-end.
        frac = min(1.0, episode / args.eps_decay)
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

        total_reward = 0.0
        for _ in range(args.max_steps):
            mask = env.valid_action_mask()
            action = agent.select_action(obs, mask, epsilon)

            next_obs, reward, done, info = env.step(action)
            next_mask = env.valid_action_mask()

            agent.memory.push(
                encode_state(obs),
                action,
                reward,
                encode_state(next_obs),
                done,
                next_mask,
            )
            agent.learn()

            obs = next_obs
            total_reward += reward
            if done:
                break

        recent_wins.append(1.0 if env.won else 0.0)
        recent_rewards.append(total_reward)

        if episode % args.report_every == 0:
            win_rate = 100.0 * np.mean(recent_wins)
            avg_reward = np.mean(recent_rewards)
            print(
                f"Episode {episode:6d} | "
                f"win rate (last {len(recent_wins)}): {win_rate:5.1f}% | "
                f"avg reward: {avg_reward:6.2f} | "
                f"epsilon: {epsilon:.3f}"
            )
            log_rows.append([episode, win_rate, avg_reward, epsilon])

    agent.save(args.model_out)
    with open(args.log_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "win_rate_pct", "avg_reward", "epsilon"])
        writer.writerows(log_rows)

    print(f"\nDone. Model saved to {args.model_out}, log saved to {args.log_out}.")


if __name__ == "__main__":
    main()
