# Minesweeper AI — Reinforcement Learning from Scratch

An AI that learns to play Minesweeper by playing it thousands of times. It starts
out clicking randomly and gradually teaches itself which cells are safe, measured
by a win rate that climbs as training goes on.

This is a complete, runnable project built around a **Deep Q-Network (DQN)** with a
**convolutional neural network** — the standard modern approach for learning to act
in a grid world.

---

## 1. What's in the box

| File | What it does |
|------|--------------|
| `minesweeper.py` | The game itself. A clean Minesweeper engine with a Gym-style `reset()` / `step()` interface so an AI can play it. |
| `dqn_agent.py` | The "brain". The neural network, the experience-replay memory, and the learning logic. |
| `train.py` | Runs training: the agent plays many games and improves. Saves the model and a CSV log of its progress. |
| `play.py` | Watch the trained agent play move-by-move, or benchmark its win rate over many games. |
| `requirements.txt` | The two libraries you need (`numpy`, `torch`). |

---

## 2. Quick start

```bash
# 1. Install the dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Train on a small, fast board (great for a first run)
python train.py --rows 6 --cols 6 --mines 6 --episodes 30000

# 3. Watch your trained AI play one game move-by-move
python play.py --rows 6 --cols 6 --mines 6 --watch

# 4. Benchmark its win rate over 1000 games
python play.py --rows 6 --cols 6 --mines 6 --games 1000
```

> **Important:** the board settings (`--rows`, `--cols`, `--mines`) must match
> between `train.py` and `play.py`, because the network is shaped to a specific
> board size.

**Speed tip:** training is much faster on a board you can win quickly. Start
small. A 6×6 board with 6 mines trains in minutes on a laptop CPU; the full
9×9-with-10-mines "Beginner" board takes considerably longer. Lower `--batch`
(e.g. `--batch 64`) speeds up each step on a CPU at a small cost to stability.

---

## 3. How it actually works (the AI part)

This is the part worth understanding, because it's what you'll talk about when
someone asks about the project.

### The core idea: Q-values

For any board, we want to answer: *"if I click this cell, how much total reward
should I expect over the rest of the game?"* That expected-future-reward number is
called a **Q-value**. If we can estimate a Q-value for every cell, playing is easy:
click the hidden cell with the highest Q-value.

We can't compute Q-values by hand, so we **approximate them with a neural
network** — that's the "Deep" in Deep Q-Network.

### Why a convolutional network

Whether a cell is safe depends entirely on the numbers immediately around it.
Convolutional layers are designed to scan local neighbourhoods on a grid, so the
network can learn a single rule like *"a revealed 1 with exactly one hidden
neighbour means that neighbour is a mine"* and apply it anywhere on the board. The
network reads the board and outputs one Q-value per cell.

### What the AI sees (state encoding)

The agent never sees the mines — only what a human sees: hidden cells and revealed
numbers. We feed the board to the network as **10 separate 0/1 layers** ("one-hot"
encoding): one layer marks hidden cells, and one layer each for the numbers 0–8.
This is better than feeding raw numbers, which would wrongly imply an "8" is eight
times more important than a "1".

### How it learns (the training loop)

The agent improves through four ideas that together make up DQN:

1. **Exploration vs. exploitation (epsilon-greedy).** Early on, the agent mostly
   clicks randomly (`epsilon` near 1.0) to discover what happens. Over training,
   `epsilon` decays toward 0, so it increasingly trusts its own learned judgment.
   This is why the win rate is *low at first and climbs* — early randomness is
   deliberate.

2. **Experience replay.** Every move (state, action, reward, next state) is stored
   in a memory buffer. Training samples a *random batch* from this memory instead
   of only the latest move. Mixing old and new experiences stops the network from
   over-fitting to whatever just happened and makes learning far more stable.

3. **The Bellman update.** The network is nudged so its prediction `Q(state, cell)`
   moves toward `reward + γ · (best Q-value of the next state)`. In plain terms:
   *the value of a move should equal the reward you got plus the value of the
   position it leaves you in.* `γ` (gamma, 0.95 here) controls how much future
   reward counts versus immediate reward.

4. **A target network.** Chasing a target that you're also changing is unstable, so
   we keep a second, slowly-updated copy of the network to provide those "next
   state" value estimates. It's refreshed periodically rather than every step.

One more practical detail: **action masking.** The agent is never allowed to click
an already-revealed cell — we force those Q-values to negative infinity before
choosing. This keeps it making real progress instead of wasting moves.

### The reward scheme

| Event | Reward |
|-------|--------|
| Win the game (all safe cells revealed) | **+1.0** |
| Hit a mine (game over) | **−1.0** |
| Reveal one or more new safe cells | **+0.3** |
| Click an already-revealed cell | **−0.3** |

These small per-move rewards matter: if the agent were *only* rewarded for the
final win, it would almost never stumble onto a win by chance and would have
nothing to learn from. Rewarding safe progress gives it a steady learning signal.

---

## 4. What to expect — and the honest limitation

You will see the win rate climb during training and then **plateau**. That plateau
is not a bug — it's a real property of Minesweeper, and it's the single best thing
you can mention to show you understand the problem:

> **Minesweeper has irreducible luck.** Some board positions genuinely cannot be
> solved by logic — two hidden cells where exactly one is a mine and no available
> information distinguishes them. The best possible player still has to guess and
> still loses a fraction of those games. So a "perfect" agent does **not** reach
> 100%; it reaches the ceiling set by how often the board forces a guess (smaller
> boards and fewer mines have a higher ceiling).

Rough expectations on a laptop CPU (your numbers will vary):

- **4×4, 3 mines:** trains in seconds, can exceed ~85% win rate. Good smoke test.
- **6×6, 6 mines:** trains in minutes, lands in roughly the ~50–70% range.
- **9×9, 10 mines (Beginner):** the real challenge; needs a long training run and
  a published research baseline is in the ~80–90% range with heavier tuning.

---

## 5. Reading your results

`train.py` writes `training_log.csv` with columns `episode, win_rate_pct,
avg_reward, epsilon`. Open it in a spreadsheet and plot `win_rate_pct` against
`episode` — that rising curve is the visual proof your AI learned, and it makes a
great screenshot for a portfolio or a slide.

---

## 6. Tuning knobs (all optional)

Command-line flags on `train.py`:

- `--episodes` — more games = more learning (and more time).
- `--batch` — replay batch size. Smaller is faster per step on CPU; larger is more
  stable.
- `--eps-decay` — over how many games exploration fades out. Larger = explores
  longer before settling.
- `--mines` — difficulty. Fewer mines is easier and trains faster.

Inside `dqn_agent.py` you can also adjust the learning rate, `gamma`, buffer size,
and how often the target network syncs.

---


## 7. Next steps / extensions (good portfolio bonuses)

- Plot the learning curve and add it to your README.
- Add a simple GUI (e.g. `pygame` or `tkinter`) so you can play against / alongside
  the AI.
- Try **Double DQN** or **Dueling DQN** — small changes to `dqn_agent.py` that are
  great talking points.
- Train on the full 9×9 Beginner board and report how close you get to published
  baselines.
