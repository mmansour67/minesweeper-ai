"""
play_human.py
=============
Play Minesweeper yourself in a graphical window -- with the trained AI riding
along as your assistant. You make the moves with the mouse; press a key and the
AI shows you the move *it* would make, or let it take a turn for you.

Run it (board size should match a model you've trained, if you want hints):
    python play_human.py --rows 6 --cols 6 --mines 6

Mouse:
    Left click    reveal a cell
    Right click   place / remove a flag (your own marker; doesn't affect the AI)

Keys:
    H    Hint - highlight the cell the AI thinks is safest to click next
    A    let the AI make ONE move for you
    R    restart with a new board
    ESC / Q   quit

Hints need a trained model file (default: minesweeper_dqn.pt) whose board size
matches. No model yet? You can still play on your own -- hints just stay off.

Needs pygame:  pip install pygame  (already in requirements.txt)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pygame

from minesweeper import Minesweeper
from dqn_agent import DQNAgent


NUMBER_COLORS = {
    1: (25, 118, 210), 2: (56, 142, 60), 3: (211, 47, 47), 4: (123, 31, 162),
    5: (255, 143, 0), 6: (0, 151, 167), 7: (66, 66, 66), 8: (158, 158, 158),
}
HIDDEN_COLOR = (150, 150, 162)
HIDDEN_HOVER = (170, 170, 182)
REVEALED_COLOR = (228, 228, 232)
MINE_COLOR = (211, 47, 47)
FLAG_COLOR = (230, 124, 34)
HINT_COLOR = (255, 213, 0)
GRID_COLOR = (105, 105, 115)
BG_COLOR = (38, 38, 46)
TEXT_COLOR = (236, 236, 240)


def ai_best_action(agent: DQNAgent, env: Minesweeper):
    """Return the flat index of the cell the AI would click next, or None."""
    if agent is None or env.done:
        return None
    obs = env.observation()
    mask = env.valid_action_mask()
    if not mask.any():
        return None
    return agent.select_action(obs, mask, epsilon=0.0)


def draw(screen, env, font, big, cell, top, flags, hint, status):
    screen.fill(BG_COLOR)
    for r in range(env.rows):
        for c in range(env.cols):
            rect = pygame.Rect(c * cell, top + r * cell, cell, cell)
            revealed = env.revealed[r, c]
            is_mine = (r, c) in env.mines
            lost = env.done and not env.won

            if revealed:
                pygame.draw.rect(screen, REVEALED_COLOR, rect)
                n = int(env.counts[r, c])
                if n > 0:
                    label = big.render(str(n), True, NUMBER_COLORS.get(n, (0, 0, 0)))
                    screen.blit(label, label.get_rect(center=rect.center))
            elif lost and is_mine:
                pygame.draw.rect(screen, MINE_COLOR, rect)
                label = big.render("*", True, (255, 255, 255))
                screen.blit(label, label.get_rect(center=rect.center))
            else:
                pygame.draw.rect(screen, HIDDEN_COLOR, rect)
                if (r, c) in flags:
                    label = big.render("F", True, FLAG_COLOR)
                    screen.blit(label, label.get_rect(center=rect.center))

            # Highlight the AI's suggested move.
            if hint is not None and hint == (r, c):
                pygame.draw.rect(screen, HINT_COLOR, rect, 4)
            else:
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

    bar = font.render(status, True, TEXT_COLOR)
    screen.blit(bar, (8, 10))
    pygame.display.flip()


def cell_from_pos(pos, cell, top, rows, cols):
    x, y = pos
    if y < top:
        return None
    r, c = (y - top) // cell, x // cell
    if 0 <= r < rows and 0 <= c < cols:
        return int(r), int(c)
    return None


def main():
    parser = argparse.ArgumentParser(description="Play Minesweeper with an AI assistant.")
    parser.add_argument("--model", type=str, default="minesweeper_dqn.pt")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--mines", type=int, default=6)
    parser.add_argument("--cell", type=int, default=56, help="tile size in pixels")
    args = parser.parse_args()

    env = Minesweeper(rows=args.rows, cols=args.cols, n_mines=args.mines)
    env.reset()

    # Load the AI assistant if a matching model exists; otherwise play unaided.
    agent = None
    if os.path.exists(args.model):
        try:
            agent = DQNAgent(rows=args.rows, cols=args.cols)
            agent.load(args.model)
            agent.policy_net.eval()
        except Exception as exc:  # wrong board size, corrupt file, etc.
            print(f"Could not load model ({exc}); playing without AI hints.")
            agent = None
    else:
        print(f"No model at '{args.model}'. Playing without AI hints "
              f"(train one with train.py to enable them).")

    pygame.init()
    top = 44
    width = args.cols * args.cell
    height = top + args.rows * args.cell
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Minesweeper -- You + AI")
    font = pygame.font.SysFont("menlo", 18)
    big = pygame.font.SysFont("menlo", max(20, args.cell // 2), bold=True)

    flags: set[tuple[int, int]] = set()
    hint = None

    def restart():
        nonlocal flags, hint
        env.reset()
        flags = set()
        hint = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    restart()
                elif event.key == pygame.K_h:
                    a = ai_best_action(agent, env)
                    hint = None if a is None else divmod(a, env.cols)
                elif event.key == pygame.K_a:
                    a = ai_best_action(agent, env)
                    if a is not None:
                        env.step(a)
                        hint = None

            elif event.type == pygame.MOUSEBUTTONDOWN and not env.done:
                cell = cell_from_pos(event.pos, args.cell, top, env.rows, env.cols)
                if cell is None:
                    continue
                r, c = cell
                if event.button == 1:  # left click = reveal
                    if cell not in flags and not env.revealed[r, c]:
                        env.step(r * env.cols + c)
                        hint = None
                elif event.button == 3:  # right click = toggle flag
                    if not env.revealed[r, c]:
                        flags.discard(cell) if cell in flags else flags.add(cell)

        # Build the status line.
        if env.done:
            outcome = "YOU WIN!  " if env.won else "BOOM - you hit a mine.  "
            status = outcome + "Press R for a new board."
        else:
            hint_txt = "H = hint   A = AI move   " if agent else "(no model: hints off)   "
            status = f"{hint_txt}R = new   flags: {len(flags)}"

        draw(screen, env, font, big, args.cell, top, flags, hint, status)
        pygame.time.wait(20)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
