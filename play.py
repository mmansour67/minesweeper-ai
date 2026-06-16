"""
play.py
=======
Watch a trained agent play Minesweeper, or benchmark its win rate over many
games. Use this to *demonstrate* what the AI learned.

Examples:
    python play.py --watch                 # show one game move-by-move
    python play.py --games 1000            # report win rate over 1000 games
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from minesweeper import Minesweeper
from dqn_agent import DQNAgent


def run_episode(env: Minesweeper, agent: DQNAgent, watch: bool, delay: float) -> bool:
    obs = env.reset()
    if watch:
        print("\n" + "=" * 40 + "\nNew game\n" + "=" * 40)
        print(env.render())
    for _ in range(env.rows * env.cols):
        mask = env.valid_action_mask()
        # epsilon=0 => always exploit (play its best move, no random exploration).
        action = agent.select_action(obs, mask, epsilon=0.0)
        obs, reward, done, info = env.step(action)
        if watch:
            print(f"\nClicked {divmod(action, env.cols)}  (reward {reward:+.1f})")
            print(env.render())
            time.sleep(delay)
        if done:
            break
    return env.won


def main():
    parser = argparse.ArgumentParser(description="Watch or benchmark a trained agent.")
    parser.add_argument("--model", type=str, default="minesweeper_dqn.pt")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--mines", type=int, default=6)
    parser.add_argument("--games", type=int, default=1000,
                        help="number of games to benchmark when not watching")
    parser.add_argument("--watch", action="store_true",
                        help="show a single game move-by-move instead of benchmarking")
    parser.add_argument("--delay", type=float, default=0.6,
                        help="seconds between moves in watch mode")
    args = parser.parse_args()

    env = Minesweeper(rows=args.rows, cols=args.cols, n_mines=args.mines)
    agent = DQNAgent(rows=args.rows, cols=args.cols)
    agent.load(args.model)
    agent.policy_net.eval()

    if args.watch:
        won = run_episode(env, agent, watch=True, delay=args.delay)
        print("\nResult:", "WIN \U0001f389" if won else "LOSS \U0001f4a5")
        return

    wins = sum(run_episode(env, agent, watch=False, delay=0.0) for _ in range(args.games))
    print(f"Won {wins}/{args.games} games  =>  win rate {100.0 * wins / args.games:.1f}%")


if __name__ == "__main__":
    main()
