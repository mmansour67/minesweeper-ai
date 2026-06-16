"""
dqn_agent.py
============
The "brain" that learns to play Minesweeper.

This implements a Deep Q-Network (DQN). The short version of the idea:

  * For any board state, we want to estimate the *value* of clicking each cell
    - i.e. "how much total future reward do I expect if I click here?".
    That estimate is called a Q-value.
  * A neural network looks at the board and outputs one Q-value per cell.
  * To play, the agent clicks the hidden cell with the highest Q-value.
  * To learn, it plays thousands of games, stores its experiences, and nudges
    the network so its Q-value predictions match what actually happened.

Why a *convolutional* network? Minesweeper is spatial: whether a cell is safe
depends on the numbers around it. Convolutions are built to read local
neighbourhoods on a grid, so the same "is this neighbourhood safe?" pattern is
recognised anywhere on the board.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# The neural network
# --------------------------------------------------------------------------- #
class MinesweeperNet(nn.Module):
    """Maps a board observation to one Q-value per cell.

    Input shape : (batch, 10, rows, cols)  -- a one-hot encoding (see encode_state)
    Output shape: (batch, rows*cols)        -- a Q-value for clicking each cell
    """

    def __init__(self, rows: int, cols: int, in_channels: int = 10):
        super().__init__()
        self.rows = rows
        self.cols = cols
        # "padding=1" keeps the board the same size after each convolution, so
        # spatial information lines up cell-for-cell from input to output.
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # A 1x1 convolution turns the 128 features at each cell into a single
        # Q-value for that cell.
        self.head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.head(x)                       # (batch, 1, rows, cols)
        return x.view(x.size(0), -1)           # flatten to (batch, rows*cols)


# --------------------------------------------------------------------------- #
# State encoding
# --------------------------------------------------------------------------- #
def encode_state(obs: np.ndarray) -> np.ndarray:
    """Turn a board observation into a one-hot tensor the network can read.

    The raw observation uses -1 for hidden and 0-8 for revealed numbers. Feeding
    those raw integers in would imply "8 is much bigger than 1", which is
    misleading. Instead we use 10 separate channels (one per possible cell
    state), each a 0/1 map. This is standard practice for categorical grids.

    Channels:
        0      -> hidden cell
        1..9   -> revealed cell showing the number (i-1), i.e. channel 1 == "0"
    """
    rows, cols = obs.shape
    planes = np.zeros((10, rows, cols), dtype=np.float32)
    hidden = obs == -1
    planes[0][hidden] = 1.0
    for n in range(0, 9):
        planes[n + 1][obs == n] = 1.0
    return planes


# --------------------------------------------------------------------------- #
# Experience replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Stores past transitions so the network can learn from a random mix of
    them. Training on a random batch (rather than only the most recent move)
    breaks correlations between consecutive frames and stabilises learning."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_masks, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class DQNAgent:
    def __init__(
        self,
        rows: int,
        cols: int,
        lr: float = 1e-3,
        gamma: float = 0.95,
        buffer_size: int = 50_000,
        batch_size: int = 256,
        target_sync: int = 1_000,
        device: str | None = None,
    ):
        self.rows = rows
        self.cols = cols
        self.n_actions = rows * cols
        self.gamma = gamma                # how much future reward counts vs. now
        self.batch_size = batch_size
        self.target_sync = target_sync    # how often to refresh the target net

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # The "online" network we train, and a slow-moving "target" copy that
        # provides stable learning targets.
        self.policy_net = MinesweeperNet(rows, cols).to(self.device)
        self.target_net = MinesweeperNet(rows, cols).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    # -- action selection --------------------------------------------------- #
    def select_action(self, obs: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy: explore randomly with probability epsilon, otherwise
        pick the highest-Q legal (hidden) cell.

        ``mask`` is a boolean array, True where a cell is still hidden. We never
        let the agent click an already-revealed cell."""
        legal = np.flatnonzero(mask)
        if random.random() < epsilon:
            return int(random.choice(legal))

        state = encode_state(obs)
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q = self.policy_net(t).cpu().numpy().flatten()
        # Block illegal cells by forcing their Q-value to -infinity.
        q[~mask] = -np.inf
        return int(np.argmax(q))

    # -- learning step ------------------------------------------------------ #
    def learn(self):
        if len(self.memory) < self.batch_size:
            return None  # not enough experience yet

        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(
            self.batch_size
        )

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        next_masks = torch.from_numpy(next_masks).to(self.device)

        # Q(s, a) for the actions we actually took.
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a'), masking illegal next moves.
        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q[~next_masks] = -np.inf            # don't value illegal moves
            max_next_q = next_q.max(dim=1).values
            # If the next state is terminal there is no future reward.
            max_next_q = torch.where(
                dones.bool(), torch.zeros_like(max_next_q), max_next_q
            )
            target = rewards + self.gamma * max_next_q

        loss = F.smooth_l1_loss(q_values, target)   # Huber loss: robust to outliers

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Periodically copy the trained weights into the target network.
        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str):
        torch.save(
            {
                "rows": self.rows,
                "cols": self.cols,
                "state_dict": self.policy_net.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["state_dict"])
        self.target_net.load_state_dict(ckpt["state_dict"])
