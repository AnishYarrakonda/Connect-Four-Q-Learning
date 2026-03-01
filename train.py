"""
train.py — curriculum DQN training against progressively stronger MCTS opponents.

Curriculum ladder:
  Stage 0 → random opponent            (depth 0)
  Stage 1 → 1-ply MCTS  (depth 1)
  Stage 2 → 2-ply MCTS  (depth 2)
  ...up to MAX_MCTS_DEPTH

Promotion: when the agent wins >= PROMOTE_WIN_RATE of the last EVAL_WINDOW
games against the current stage, it advances to the next stage.
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
# Config  — edit these directly, no interactive prompts
# ---------------------------------------------------------------------------

SAVE_DIR            = "models"
RUN_NAME            = "cf_dqn"
NUM_EPISODES        = 50_000
SAVE_INTERVAL       = 2_000       # save a checkpoint every N episodes
REPORT_INTERVAL     = 200         # print a stats line every N episodes

# Curriculum
MAX_MCTS_DEPTH      = 6           # highest MCTS depth to train against
MCTS_SIMULATIONS    = 40          # rollouts per move for the MCTS opponent
PROMOTE_WIN_RATE    = 0.60        # win-rate threshold to advance to next stage
EVAL_WINDOW         = 300         # rolling window size for win-rate evaluation

# DQN hyper-params
LR                  = 1e-3
GAMMA               = 0.99
BATCH_SIZE          = 64
BUFFER_CAPACITY     = 60_000
TARGET_UPDATE_FREQ  = 500
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY       = 12_000      # steps for epsilon annealing

# Rewards
WIN_REWARD          = 1.0
LOSS_REWARD         = -1.0
DRAW_REWARD         = 0.2         # draws are slightly positive vs losses

# Resume
RESUME_PATH         = ""          # set to a .pth path to resume, else ""

# ---------------------------------------------------------------------------

class ANSI:
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    MAGENTA= "\033[95m"
    RESET  = "\033[0m"


def _clone_board(board: Board) -> Board:
    b = Board()
    b.player1_bits = board.player1_bits.clone()
    b.player2_bits = board.player2_bits.clone()
    b.turn = board.turn
    b.move_history = board.move_history.copy()
    return b


def play_episode(
    agent: DQNAgent,
    opponent: MCTSOpponent,
    agent_is_p1: bool,
) -> tuple[int, int]:
    """
    Play one full game.  Agent plays as P1 when agent_is_p1=True, else as P2.
    Returns (winner, game_length).
    winner: 1=P1 wins, 2=P2 wins, 0=draw.
    """
    board = Board()
    trajectory: list[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = []

    agent_player   = 1 if agent_is_p1 else 2
    opp_player     = 2 if agent_is_p1 else 1

    while True:
        current_player = 1 if board.turn % 2 == 0 else 2
        valid = board.valid_moves()

        if not valid:
            winner = 0
            break

        state = Board.board_to_tensor(board)

        if current_player == agent_player:
            action = agent.select_action(board)
        else:
            action = opponent.select_action(board)

        row = board.make_move(action)
        if row is None:
            winner = opp_player  # illegal move → lose
            break

        done, winner = board.game_over(row, action)

        # Build reward and transition only for agent moves
        if current_player == agent_player:
            if done:
                if winner == agent_player:
                    reward = WIN_REWARD
                elif winner == 0:
                    reward = DRAW_REWARD
                else:
                    reward = LOSS_REWARD
            else:
                reward = 0.0

            next_state = Board.board_to_tensor(board)
            trajectory.append((state, action, reward, next_state, done))

        if done:
            break

    # If the opponent won, back-fill a loss onto the last agent transition
    if trajectory and winner == opp_player:
        s, a, _, ns, d = trajectory[-1]
        trajectory[-1] = (s, a, LOSS_REWARD, ns, True)

    # Push all agent transitions into replay and run one learn step
    for (s, a, r, ns, d) in trajectory:
        agent.buffer.push(s, a, r, ns, d)

    agent.learn()

    return winner, board.turn


def save_checkpoint(agent: DQNAgent, episode: int, stage: int) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{RUN_NAME}_ep{episode}_stage{stage}.pth")
    agent.save(path)
    return path


def print_stats(
    episode: int,
    stage: int,
    p1_wins: int,
    p2_wins: int,
    draws: int,
    rolling_win_rate: float,
    avg_len: float,
    epsilon: float,
    total_s: float,
) -> None:
    total = p1_wins + p2_wins + draws or 1
    line = (
        f"Ep {episode:>6} | "
        f"Stage {ANSI.CYAN}{stage}{ANSI.RESET} | "
        f"P1 {ANSI.RED}{p1_wins:>5}{ANSI.RESET} "
        f"P2 {ANSI.YELLOW}{p2_wins:>5}{ANSI.RESET} "
        f"D {ANSI.CYAN}{draws:>4}{ANSI.RESET} | "
        f"WinRate {ANSI.GREEN}{rolling_win_rate:.2%}{ANSI.RESET} | "
        f"AvgLen {avg_len:>5.1f} | "
        f"ε {ANSI.MAGENTA}{epsilon:.3f}{ANSI.RESET} | "
        f"Time {total_s:>7.1f}s"
    )
    print(line)


def run_training() -> None:
    print(f"Starting curriculum DQN training — {NUM_EPISODES} episodes")
    print(f"MCTS ladder: depth 0 → {MAX_MCTS_DEPTH}  |  promote at {PROMOTE_WIN_RATE:.0%} over {EVAL_WINDOW} games\n")

    agent = DQNAgent(
        lr=LR,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
    )

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        agent.load(RESUME_PATH)
        print(f"Resumed from {RESUME_PATH}")

    current_stage = 0
    opponent = MCTSOpponent(depth=current_stage, n_simulations=MCTS_SIMULATIONS)

    p1_wins = draws = p2_wins = 0
    rolling: list[int] = []   # 1=agent win, 0=draw/loss
    game_lengths: list[int] = []
    total_start = time.perf_counter()

    for episode in range(1, NUM_EPISODES + 1):
        # Alternate sides every episode for balanced training
        agent_is_p1 = (episode % 2 == 1)

        winner, length = play_episode(agent, opponent, agent_is_p1)
        game_lengths.append(length)

        agent_player = 1 if agent_is_p1 else 2
        if winner == agent_player:
            p1_wins += 1
            rolling.append(1)
        elif winner == 0:
            draws += 1
            rolling.append(0)
        else:
            p2_wins += 1
            rolling.append(0)

        if len(rolling) > EVAL_WINDOW:
            rolling.pop(0)

        rolling_win_rate = sum(rolling) / len(rolling) if rolling else 0.0

        # Curriculum promotion
        if (
            len(rolling) >= EVAL_WINDOW
            and rolling_win_rate >= PROMOTE_WIN_RATE
            and current_stage < MAX_MCTS_DEPTH
        ):
            current_stage += 1
            opponent = MCTSOpponent(depth=current_stage, n_simulations=MCTS_SIMULATIONS)
            rolling.clear()
            save_checkpoint(agent, episode, current_stage - 1)
            print(f"\n{'='*70}")
            print(f"  ✓ Promoted to Stage {current_stage}  (MCTS depth {current_stage})")
            print(f"{'='*70}\n")

        # Periodic reporting
        if episode % REPORT_INTERVAL == 0:
            avg_len = statistics.mean(game_lengths[-REPORT_INTERVAL:])
            print_stats(
                episode=episode,
                stage=current_stage,
                p1_wins=p1_wins,
                p2_wins=p2_wins,
                draws=draws,
                rolling_win_rate=rolling_win_rate,
                avg_len=avg_len,
                epsilon=agent.epsilon,
                total_s=time.perf_counter() - total_start,
            )

        # Periodic checkpointing
        if episode % SAVE_INTERVAL == 0:
            path = save_checkpoint(agent, episode, current_stage)
            print(f"  → Checkpoint: {path}")

    # Final save
    final_path = os.path.join(SAVE_DIR, f"{RUN_NAME}_final.pth")
    agent.save(final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Total time: {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    run_training()
