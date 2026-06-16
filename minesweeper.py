"""
minesweeper.py
==============
A self-contained Minesweeper environment designed for reinforcement learning.

The class below behaves a bit like an OpenAI-Gym environment: you call
``reset()`` to start a new game and ``step(action)`` to take a move. Each step
returns the new observation, a reward, and whether the game is over. That is the
exact interface a learning agent needs.

Board encoding used for the *observation* (what the AI "sees"):
    -1   : unrevealed / hidden cell
     0-8 : a revealed cell showing how many mines touch it

Note the AI never sees where the mines are. It only sees revealed numbers and
hidden cells, exactly like a human player.
"""

from __future__ import annotations

import random
import numpy as np


# Values used internally on the observation grid.
HIDDEN = -1


class Minesweeper:
    def __init__(self, rows: int = 9, cols: int = 9, n_mines: int = 10, seed: int | None = None):
        self.rows = rows
        self.cols = cols
        self.n_mines = n_mines
        self.rng = random.Random(seed)

        # These are filled in by reset().
        self.mines: set[tuple[int, int]] = set()
        self.counts = np.zeros((rows, cols), dtype=np.int8)   # true neighbour-mine counts
        self.revealed = np.zeros((rows, cols), dtype=bool)     # which cells the player has opened
        self.done = False
        self.won = False
        self._mines_placed = False

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        """Start a fresh game and return the first observation."""
        self.mines = set()
        self.counts = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.revealed = np.zeros((self.rows, self.cols), dtype=bool)
        self.done = False
        self.won = False
        self._mines_placed = False
        return self.observation()

    def _place_mines(self, safe: tuple[int, int]) -> None:
        """Place mines randomly, guaranteeing the first clicked cell is safe.

        Mines are placed *after* the first click (a standard Minesweeper rule)
        so the player never loses on move one.
        """
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        # Keep the first-clicked cell (and ideally its neighbours) mine-free.
        forbidden = set(self._neighbours(*safe)) | {safe}
        candidates = [cell for cell in all_cells if cell not in forbidden]

        # If the board is tiny and we can't spare the neighbours, fall back to
        # only protecting the clicked cell itself.
        if len(candidates) < self.n_mines:
            candidates = [cell for cell in all_cells if cell != safe]

        self.mines = set(self.rng.sample(candidates, self.n_mines))

        # Pre-compute the neighbour counts once mines are known.
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.mines:
                    self.counts[r, c] = -1  # marker, never shown to the player
                else:
                    self.counts[r, c] = sum(
                        1 for nr, nc in self._neighbours(r, c) if (nr, nc) in self.mines
                    )
        self._mines_placed = True

    def _neighbours(self, r: int, c: int):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield nr, nc

    # ------------------------------------------------------------------ #
    # Core gameplay
    # ------------------------------------------------------------------ #
    def _flood_reveal(self, r: int, c: int) -> int:
        """Reveal a cell; if it is a 0 the empty region cascades open.

        Returns the number of newly revealed cells.
        """
        stack = [(r, c)]
        newly = 0
        while stack:
            cr, cc = stack.pop()
            if self.revealed[cr, cc]:
                continue
            self.revealed[cr, cc] = True
            newly += 1
            # A "0" cell has no adjacent mines, so its neighbours are all safe
            # and get opened automatically.
            if self.counts[cr, cc] == 0:
                for nr, nc in self._neighbours(cr, cc):
                    if not self.revealed[nr, nc]:
                        stack.append((nr, nc))
        return newly

    def step(self, action: int):
        """Apply an action and return (observation, reward, done, info).

        ``action`` is a flat index in range [0, rows*cols). It is decoded into
        the (row, col) cell the agent wants to reveal.

        Reward scheme (tuned so the agent learns to make safe progress):
            +1.0  win the game (all safe cells revealed)
            -1.0  hit a mine (game over)
            +0.3  revealed one or more new safe cells (good progress)
            -0.3  clicked an already-revealed cell (wasted, illegal-ish move)
        """
        if self.done:
            raise RuntimeError("step() called on a finished game; call reset() first.")

        r, c = divmod(action, self.cols)

        # First click: place mines now so the opening move is always safe.
        if not self._mines_placed:
            self._place_mines((r, c))

        # Clicking a cell that is already open wastes a turn.
        if self.revealed[r, c]:
            return self.observation(), -0.3, False, {"result": "no_progress"}

        # Clicking a mine ends the game.
        if (r, c) in self.mines:
            self.done = True
            self.won = False
            return self.observation(), -1.0, True, {"result": "loss"}

        # Otherwise reveal the cell (and any cascade).
        newly = self._flood_reveal(r, c)

        # Win check: every non-mine cell has been revealed.
        safe_cells = self.rows * self.cols - self.n_mines
        if int(self.revealed.sum()) == safe_cells:
            self.done = True
            self.won = True
            return self.observation(), 1.0, True, {"result": "win"}

        reward = 0.3 if newly > 0 else -0.3
        return self.observation(), reward, False, {"result": "progress", "newly": newly}

    # ------------------------------------------------------------------ #
    # Observations and rendering
    # ------------------------------------------------------------------ #
    def observation(self) -> np.ndarray:
        """Return the grid the agent sees: -1 for hidden, 0-8 for revealed."""
        obs = np.full((self.rows, self.cols), HIDDEN, dtype=np.int8)
        revealed_idx = self.revealed
        obs[revealed_idx] = self.counts[revealed_idx]
        return obs

    def valid_action_mask(self) -> np.ndarray:
        """Boolean flat array: True where a cell is still hidden (legal to click)."""
        return (~self.revealed).flatten()

    def render(self) -> str:
        """Return a human-readable string of the current board state."""
        symbols = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if not self.revealed[r, c]:
                    # Show mines only after the game is lost.
                    if self.done and not self.won and (r, c) in self.mines:
                        row.append("*")
                    else:
                        row.append(".")
                else:
                    n = int(self.counts[r, c])
                    row.append(str(n) if n > 0 else " ")
            symbols.append(" ".join(row))
        header = "    " + " ".join(str(c % 10) for c in range(self.cols))
        body = "\n".join(f"{r % 10:2d}  {line}" for r, line in enumerate(symbols))
        return header + "\n" + body


if __name__ == "__main__":
    # Quick manual sanity check: play a few random moves and print the board.
    env = Minesweeper(rows=9, cols=9, n_mines=10, seed=0)
    obs = env.reset()
    print("Fresh board:")
    print(env.render())
    for _ in range(5):
        mask = env.valid_action_mask()
        legal = np.flatnonzero(mask)
        action = int(np.random.choice(legal))
        obs, reward, done, info = env.step(action)
        print(f"\nAction {divmod(action, env.cols)} -> reward {reward}, info {info}")
        print(env.render())
        if done:
            print("\nGame over:", "WIN" if env.won else "LOSS")
            break
