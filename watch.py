"""
watch.py
========
A graphical window where you watch the trained AI play Minesweeper, one move at
a time, on a real coloured board. This is the fun one -- and it makes a great
screen-recording or screenshot for a portfolio.

Run it (board size must match what you trained on):
    python watch.py --rows 6 --cols 6 --mines 6
    python watch.py --rows 4 --cols 4 --mines 3 --delay 0.4

Controls:
    SPACE  pause / resume
    N      skip to the next game immediately
    ESC/Q  quit

Needs pygame:  pip install pygame  (already in requirements.txt)
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pygame

from minesweeper import Minesweeper
from dqn_agent import DQNAgent


# Classic Minesweeper number colours (1-8).
NUMBER_COLORS = {
    1: (25, 118, 210),    # blue
    2: (56, 142, 60),     # green
    3: (211, 47, 47),     # red
    4: (123, 31, 162),    # purple
    5: (255, 143, 0),     # orange
    6: (0, 151, 167),     # teal
    7: (66, 66, 66),      # dark grey
    8: (158, 158, 158),   # grey
}

HIDDEN_COLOR = (160, 160, 170)
REVEALED_COLOR = (225, 225, 230)
MINE_COLOR = (211, 47, 47)
GRID_COLOR = (110, 110, 120)
BG_COLOR = (40, 40, 48)
TEXT_COLOR = (235, 235, 240)


def draw_board(screen, env, font, cell, top, status):
    screen.fill(BG_COLOR)
    for r in range(env.rows):
        for c in range(env.cols):
            x, y = c * cell, top + r * cell
            rect = pygame.Rect(x, y, cell, cell)

            revealed = env.revealed[r, c]
            is_mine = (r, c) in env.mines
            lost = env.done and not env.won

            if revealed:
                pygame.draw.rect(screen, REVEALED_COLOR, rect)
                n = int(env.counts[r, c])
                if n > 0:
                    label = font.render(str(n), True, NUMBER_COLORS.get(n, (0, 0, 0)))
                    screen.blit(label, label.get_rect(center=rect.center))
            elif lost and is_mine:
                # Reveal the mines once the game is lost.
                pygame.draw.rect(screen, MINE_COLOR, rect)
                label = font.render("*", True, (255, 255, 255))
                screen.blit(label, label.get_rect(center=rect.center))
            else:
                pygame.draw.rect(screen, HIDDEN_COLOR, rect)

            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

    # Status bar across the top.
    bar = font.render(status, True, TEXT_COLOR)
    screen.blit(bar, (8, 8))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Watch the trained AI play Minesweeper.")
    parser.add_argument("--model", type=str, default="minesweeper_dqn.pt")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--mines", type=int, default=6)
    parser.add_argument("--delay", type=float, default=0.5,
                        help="seconds between moves (lower = faster)")
    parser.add_argument("--cell", type=int, default=64, help="tile size in pixels")
    parser.add_argument("--games", type=int, default=0,
                        help="auto-quit after this many games (0 = run forever)")
    args = parser.parse_args()

    env = Minesweeper(rows=args.rows, cols=args.cols, n_mines=args.mines)
    agent = DQNAgent(rows=args.rows, cols=args.cols)
    agent.load(args.model)
    agent.policy_net.eval()

    pygame.init()
    top = 36  # height of the status bar
    width = args.cols * args.cell
    height = top + args.rows * args.cell
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Minesweeper AI")
    font = pygame.font.SysFont("menlo", max(18, args.cell // 3))

    obs = env.reset()
    wins = games = 0
    paused = False
    last_move = time.time()
    move_in_game = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    obs = env.reset()
                    move_in_game = 0

        win_pct = (100.0 * wins / games) if games else 0.0
        status = (f"Game {games + 1}   moves {move_in_game}   "
                  f"won {wins}/{games} ({win_pct:.0f}%)   "
                  f"{'PAUSED' if paused else ''}")
        draw_board(screen, env, font, args.cell, top, status)

        now = time.time()
        if not paused and not env.done and (now - last_move) >= args.delay:
            mask = env.valid_action_mask()
            action = agent.select_action(obs, mask, epsilon=0.0)  # always its best move
            obs, _, _, _ = env.step(action)
            move_in_game += 1
            last_move = now

        if env.done:
            # Briefly show the finished board, then start a new game.
            draw_board(screen, env, font, args.cell, top, status)
            time.sleep(max(args.delay, 0.8))
            games += 1
            if env.won:
                wins += 1
            if args.games and games >= args.games:
                running = False
            else:
                obs = env.reset()
                move_in_game = 0
                last_move = time.time()

        pygame.time.wait(20)  # keep the UI responsive without burning CPU

    pygame.quit()
    if games:
        print(f"Watched {games} games, AI won {wins} ({100.0 * wins / games:.1f}%).")
    sys.exit(0)


if __name__ == "__main__":
    main()
