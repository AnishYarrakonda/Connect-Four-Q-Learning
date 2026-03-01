"""
train.py — curriculum DQN training against progressively stronger MCTS opponents.

Curriculum ladder:
  Stage 0 → random opponent   (depth 0)
  Stage 1 → 1-ply MCTS        (depth 1)
  ...up to MAX_MCTS_DEPTH

Promotion rule (simple):
  Every EVAL_INTERVAL training episodes, play EVAL_GAMES greedy games where
  each game starts from a random board position (4–20 moves already played).
  Agent must win >= EVAL_WIN_THRESHOLD of those to advance to the next stage.
"""

import os
import time
import random
import statistics

import torch

from board import Board
from agent import DQNAgent
from mcts import MCTSOpponent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAVE_DIR        = "models"
RUN_NAME        = "cf_dqn"
NUM_EPISODES    = 50_000
SAVE_INTERVAL   = 5_000
REPORT_INTERVAL = 250

# Curriculum
MAX_MCTS_DEPTH  = 6
MCTS_SIMS       = 20          # rollouts per move — reduced from 40, still strong

# Promotion  — every 1000 games, 200 greedy games from random positions, need 140 wins
EVAL_INTERVAL       = 1000
EVAL_GAMES          = 200
EVAL_WIN_THRESHOLD  = 140     # 140/200 = 70%
EVAL_START_MOVES    = (4, 20) # random starting position: 4 to 20 random moves pre-played

# DQN hyper-params
LR              = 2e-3
GAMMA           = 0.99
BATCH_SIZE      = 64
BUFFER_CAPACITY = 60_000
TARGET_UPDATE   = 500
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 12_000

# Rewards
WIN_REWARD  =  1.0
LOSS_REWARD = -1.0
DRAW_REWARD =  0.2

RESUME_PATH = ""   # path to .pth to resume from, or ""

# ---------------------------------------------------------------------------

class ANSI:
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    MAGENTA= "\033[95m"
    RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Training episode
# ---------------------------------------------------------------------------

def play_episode(agent: DQNAgent, opponent: MCTSOpponent, agent_is_p1: bool) -> tuple[int, int]:
    board        = Board()
    agent_player = 1 if agent_is_p1 else 2
    opp_player   = 2 if agent_is_p1 else 1
    trajectory   = []

    while True:
        current = 1 if board.turn % 2 == 0 else 2
        valid   = board.valid_moves()
        if not valid:
            winner = 0
            break

        state  = Board.board_to_tensor(board)
        action = agent.select_action(board) if current == agent_player else opponent.select_action(board)

        row = board.make_move(action)
        if row is None:
            winner = opp_player
            break

        done, winner = board.game_over(row, action)

        if current == agent_player:
            if done:
                reward = WIN_REWARD if winner == agent_player else (DRAW_REWARD if winner == 0 else LOSS_REWARD)
            else:
                reward = 0.0
            trajectory.append((state, action, reward, Board.board_to_tensor(board), done))

        if done:
            break

    if trajectory and winner == opp_player:
        s, a, _, ns, d = trajectory[-1]
        trajectory[-1] = (s, a, LOSS_REWARD, ns, True)

    for t in trajectory:
        agent.buffer.push(*t)
    agent.learn()

    return winner, board.turn


# ---------------------------------------------------------------------------
# Eval — greedy games from random mid-game starting positions
# Runs entirely on FastBoard — zero PyTorch tensor ops during eval
# ---------------------------------------------------------------------------

def _random_start_fastboard(min_moves: int, max_moves: int) -> "FastBoard":
    """Build a random mid-game FastBoard with min_moves..max_moves already played."""
    from mcts import FastBoard
    while True:
        fb = FastBoard()
        n  = random.randint(min_moves, max_moves)
        for _ in range(n):
            valid = fb.valid_moves()
            if not valid:
                break
            col = random.choice(valid)
            r   = fb.make_move(col)
            if fb.check_win(fb.last_player()):
                fb = FastBoard()   # position already terminal — start over
                break
        else:
            return fb   # successfully built a non-terminal position


def _fastboard_to_board(fb: "FastBoard") -> Board:
    """Convert a FastBoard back to a tensor Board for agent inference."""
    from mcts import ROWS, COLS
    board = Board()
    for r in range(ROWS):
        for c in range(COLS):
            val = fb.cells[r * COLS + c]
            if val == 1:
                board.player1_bits[r, c] = 1.0
            elif val == 2:
                board.player2_bits[r, c] = 1.0
    board.turn = fb.turn
    return board


def run_eval(agent: DQNAgent, opponent: MCTSOpponent) -> tuple[int, int, int]:
    """
    EVAL_GAMES greedy games (ε=0) from random mid-game positions.
    Alternates sides each game. No buffer pushes, no learning.
    Returns (wins, draws, losses).

    Board state is kept as FastBoard throughout; only converted to tensor
    once per agent move (for inference). The opponent never touches tensors.
    """
    wins = draws = losses = 0

    for i in range(EVAL_GAMES):
        fb           = _random_start_fastboard(*EVAL_START_MOVES)
        agent_is_p1  = (i % 2 == 0)
        agent_player = 1 if agent_is_p1 else 2

        while True:
            valid = fb.valid_moves()
            if not valid:
                winner = 0
                break

            current = (fb.turn % 2) + 1

            if current == agent_player:
                # One tensor conversion per agent move — unavoidable for inference
                board  = _fastboard_to_board(fb)
                action = agent.policy_net.best_move(board)
            else:
                # Opponent works entirely on FastBoard — no tensor ops
                action = opponent.select_action_fast(fb)

            r = fb.make_move(action)
            if r < 0:
                winner = 3 - agent_player
                break

            last = fb.last_player()
            if fb.check_win(last):
                winner = last
                break
            if fb.is_full():
                winner = 0
                break

        if winner == agent_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(agent: DQNAgent, episode: int, stage: int) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{RUN_NAME}_ep{episode}_stage{stage}.pth")
    agent.save(path)
    return path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_training() -> None:
    print(f"Curriculum DQN — {NUM_EPISODES} episodes | MCTS depth 0→{MAX_MCTS_DEPTH}")
    print(f"Promotion: every {EVAL_INTERVAL} eps, {EVAL_WIN_THRESHOLD}/{EVAL_GAMES} wins "
          f"(greedy, ε=0, from {EVAL_START_MOVES[0]}–{EVAL_START_MOVES[1]}-move random starts)\n")

    agent = DQNAgent(
        lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE,
        epsilon_start=EPS_START, epsilon_end=EPS_END, epsilon_decay=EPS_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
    )

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        agent.load(RESUME_PATH)
        print(f"Resumed from {RESUME_PATH}")

    stage    = 0
    opponent = MCTSOpponent(depth=stage, n_simulations=MCTS_SIMS)

    w_wins = w_losses = w_draws = 0   # per-window counters, reset every REPORT_INTERVAL
    game_lengths = []
    t0 = time.perf_counter()

    for ep in range(1, NUM_EPISODES + 1):
        agent_is_p1 = (ep % 2 == 1)
        winner, length = play_episode(agent, opponent, agent_is_p1)
        game_lengths.append(length)

        ap = 1 if agent_is_p1 else 2
        if winner == ap:   w_wins   += 1
        elif winner == 0:  w_draws  += 1
        else:              w_losses += 1

        # Training report — last REPORT_INTERVAL games only
        if ep % REPORT_INTERVAL == 0:
            total   = w_wins + w_losses + w_draws or 1
            avg_len = statistics.mean(game_lengths[-REPORT_INTERVAL:])
            print(
                f"Ep {ep:>6} | Stage {ANSI.CYAN}{stage}{ANSI.RESET} | "
                f"W {ANSI.GREEN}{w_wins:>3}{ANSI.RESET} "
                f"L {ANSI.RED}{w_losses:>3}{ANSI.RESET} "
                f"D {ANSI.CYAN}{w_draws:>2}{ANSI.RESET} "
                f"/{REPORT_INTERVAL} "
                f"({ANSI.YELLOW}{w_wins/total:.1%} ε-train{ANSI.RESET}) | "
                f"AvgLen {avg_len:>4.1f} | "
                f"ε {ANSI.MAGENTA}{agent.epsilon:.3f}{ANSI.RESET} | "
                f"{time.perf_counter()-t0:>6.1f}s"
            )
            w_wins = w_losses = w_draws = 0   # reset for next window

        # Promotion eval
        if ep % EVAL_INTERVAL == 0 and stage < MAX_MCTS_DEPTH:
            ew, ed, el = run_eval(agent, opponent)
            promoted   = ew >= EVAL_WIN_THRESHOLD
            tag = f"{ANSI.GREEN}✓ PROMOTE{ANSI.RESET}" if promoted else f"{ANSI.RED}not yet{ANSI.RESET}"
            print(
                f"  EVAL | Stage {ANSI.CYAN}{stage}{ANSI.RESET} | "
                f"W {ANSI.GREEN}{ew}{ANSI.RESET} "
                f"D {ANSI.CYAN}{ed}{ANSI.RESET} "
                f"L {ANSI.RED}{el}{ANSI.RESET} "
                f"({ew}/{EVAL_GAMES} greedy, need {EVAL_WIN_THRESHOLD}) → {tag}"
            )
            if promoted:
                save_checkpoint(agent, ep, stage)
                stage   += 1
                opponent = MCTSOpponent(depth=stage, n_simulations=MCTS_SIMS)
                w_wins = w_losses = w_draws = 0
                game_lengths.clear()
                print(f"  → Stage {stage} (MCTS depth {stage})\n")

        # Periodic checkpoint
        if ep % SAVE_INTERVAL == 0:
            print(f"  → Checkpoint: {save_checkpoint(agent, ep, stage)}")

    agent.save(os.path.join(SAVE_DIR, f"{RUN_NAME}_final.pth"))
    print(f"\nDone. Total time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run_training()