"""
train.py — curriculum DQN training against progressively stronger MCTS opponents.

Curriculum ladder:
  Stage 0 → random opponent            (depth 0)
  Stage 1 → 1-ply MCTS  (depth 1)
  Stage 2 → 2-ply MCTS  (depth 2)
  ...up to MAX_MCTS_DEPTH

Training:   epsilon-greedy, pushes to replay buffer, runs learn() each episode.
Evaluation: fully greedy (epsilon=0), no buffer pushes, no learning.
            Run every EVAL_INTERVAL training episodes.
            Promotion is decided ONLY from eval results — epsilon never taints it.
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
SAVE_INTERVAL       = 2_000       # save a checkpoint every N training episodes
REPORT_INTERVAL     = 200         # print a training stats line every N episodes

# Curriculum
MAX_MCTS_DEPTH      = 6           # highest MCTS depth to train against
MCTS_SIMULATIONS    = 40          # rollouts per move for the MCTS opponent
PROMOTE_WIN_RATE    = 0.60        # greedy win-rate needed to advance

# Evaluation — low-temperature sampling, no learning, no epsilon
EVAL_INTERVAL       = 500         # run an eval every N training episodes
EVAL_GAMES          = 250         # games per eval — half as P1, half as P2
                                  # need round(0.60 * 250) = 150 wins to promote
EVAL_TEMPERATURE    = 0.2         # agent uses temperature sampling during eval
                                  # low enough to be near-greedy, high enough that
                                  # games aren't identical clones of each other

# Training visibility
ROLLING_WINDOW      = 250         # rolling window size shown in training stats
                                  # NOTE: epsilon-polluted — informational only, never used for promotion

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


# ---------------------------------------------------------------------------
# Single training episode — epsilon-greedy, pushes to buffer, calls learn()
# ---------------------------------------------------------------------------

def play_episode(
    agent: DQNAgent,
    opponent: MCTSOpponent,
    agent_is_p1: bool,
) -> tuple[int, int]:
    """
    Play one training game.
    Returns (winner, game_length).  winner: 1=P1, 2=P2, 0=draw.
    """
    board = Board()
    trajectory: list[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = []

    agent_player = 1 if agent_is_p1 else 2
    opp_player   = 2 if agent_is_p1 else 1

    while True:
        current_player = 1 if board.turn % 2 == 0 else 2
        valid = board.valid_moves()

        if not valid:
            winner = 0
            break

        state = Board.board_to_tensor(board)

        if current_player == agent_player:
            action = agent.select_action(board)   # epsilon-greedy
        else:
            action = opponent.select_action(board)

        row = board.make_move(action)
        if row is None:
            winner = opp_player
            break

        done, winner = board.game_over(row, action)

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

    # Back-fill loss reward if opponent won
    if trajectory and winner == opp_player:
        s, a, _, ns, d = trajectory[-1]
        trajectory[-1] = (s, a, LOSS_REWARD, ns, True)

    for (s, a, r, ns, d) in trajectory:
        agent.buffer.push(s, a, r, ns, d)

    agent.learn()

    return winner, board.turn


# ---------------------------------------------------------------------------
# Evaluation — temperature sampling, no buffer pushes, no learning
# ---------------------------------------------------------------------------

def run_eval(
    agent: DQNAgent,
    opponent: MCTSOpponent,
    n_games: int = EVAL_GAMES,
    temperature: float = EVAL_TEMPERATURE,
) -> tuple[int, int, int]:
    """
    Play n_games using low-temperature policy sampling (not pure greedy).
    Returns (wins, draws, losses) from the agent's perspective.
    Alternates sides each game so neither player has a first-move advantage.
    Does NOT touch the replay buffer or call learn().

    Why temperature and not greedy: a fully greedy agent vs a deterministic
    MCTS opponent produces the exact same game every time (2 unique games
    repeated 150x each), making the eval meaningless. Low temperature keeps
    the agent near-optimal while generating genuinely diverse game trees.
    """
    wins = draws = losses = 0

    for i in range(n_games):
        agent_is_p1  = (i % 2 == 0)
        agent_player = 1 if agent_is_p1 else 2

        board = Board()

        while True:
            current_player = 1 if board.turn % 2 == 0 else 2
            valid = board.valid_moves()

            if not valid:
                winner = 0
                break

            if current_player == agent_player:
                action = agent.policy_net.sample_move(board, temperature=temperature)
            else:
                action = opponent.select_action(board)

            row = board.make_move(action)
            if row is None:
                winner = 2 if agent_is_p1 else 1
                break

            done, winner = board.game_over(row, action)
            if done:
                break

        if winner == agent_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def print_train_stats(
    episode: int,
    stage: int,
    total_wins: int,
    total_losses: int,
    total_draws: int,
    rolling_wins: int,
    rolling_total: int,
    rolling_wr: float,
    avg_len: float,
    epsilon: float,
    elapsed_s: float,
) -> None:
    total = total_wins + total_losses + total_draws or 1
    train_wr = total_wins / total
    line = (
        f"Train Ep {episode:>6} | "
        f"Stage {ANSI.CYAN}{stage}{ANSI.RESET} | "
        f"W {ANSI.GREEN}{total_wins:>5}{ANSI.RESET} "
        f"L {ANSI.RED}{total_losses:>5}{ANSI.RESET} "
        f"D {ANSI.CYAN}{total_draws:>4}{ANSI.RESET} "
        f"(overall {train_wr:.1%}) | "
        f"Last {rolling_total}: {ANSI.YELLOW}{rolling_wins}/{rolling_total}{ANSI.RESET} "
        f"({ANSI.YELLOW}{rolling_wr:.1%} ε-polluted{ANSI.RESET}) | "
        f"AvgLen {avg_len:>5.1f} | "
        f"ε {ANSI.MAGENTA}{epsilon:.3f}{ANSI.RESET} | "
        f"Time {elapsed_s:>7.1f}s"
    )
    print(line)


def print_eval_stats(
    episode: int,
    stage: int,
    wins: int,
    draws: int,
    losses: int,
    threshold: int,
    promoted: bool,
) -> None:
    n = wins + draws + losses
    wr = wins / n if n else 0.0
    status = f"{ANSI.GREEN}✓ PROMOTED{ANSI.RESET}" if promoted else f"{ANSI.RED}not yet ({wins}/{threshold} wins){ANSI.RESET}"
    line = (
        f"  ╔═ EVAL @ ep {episode} | Stage {ANSI.CYAN}{stage}{ANSI.RESET} | "
        f"W {ANSI.GREEN}{wins}{ANSI.RESET} "
        f"D {ANSI.CYAN}{draws}{ANSI.RESET} "
        f"L {ANSI.RED}{losses}{ANSI.RESET} "
        f"({ANSI.GREEN if promoted else ANSI.YELLOW}{wr:.1%}{ANSI.RESET} eval WR @ T={EVAL_TEMPERATURE}, "
        f"need {PROMOTE_WIN_RATE:.0%}) → {status}"
    )
    print(line)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(agent: DQNAgent, episode: int, stage: int) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{RUN_NAME}_ep{episode}_stage{stage}.pth")
    agent.save(path)
    return path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training() -> None:
    promote_threshold = round(PROMOTE_WIN_RATE * EVAL_GAMES)  # 150 — use round() not int() to avoid truncation drift
    print(f"Starting curriculum DQN training — {NUM_EPISODES} episodes")
    print(f"MCTS ladder: depth 0 → {MAX_MCTS_DEPTH}")
    print(f"Promotion: greedy eval every {EVAL_INTERVAL} episodes, "
          f"need {promote_threshold}/{EVAL_GAMES} wins ({PROMOTE_WIN_RATE:.0%})")
    print(f"Training stats show a rolling {ROLLING_WINDOW}-game window (epsilon-polluted, informational only)\n")

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

    total_wins = total_losses = total_draws = 0
    rolling: list[int] = []          # 1=win, 0=draw/loss — last ROLLING_WINDOW games
    game_lengths: list[int] = []
    total_start = time.perf_counter()

    for episode in range(1, NUM_EPISODES + 1):
        agent_is_p1 = (episode % 2 == 1)   # alternates every game: P1, P2, P1, P2 ...

        winner, length = play_episode(agent, opponent, agent_is_p1)
        game_lengths.append(length)

        agent_player = 1 if agent_is_p1 else 2
        if winner == agent_player:
            total_wins += 1
            rolling.append(1)
        elif winner == 0:
            total_draws += 1
            rolling.append(0)
        else:
            total_losses += 1
            rolling.append(0)

        if len(rolling) > ROLLING_WINDOW:
            rolling.pop(0)

        rolling_wins = sum(rolling)
        rolling_wr   = rolling_wins / len(rolling) if rolling else 0.0

        # ---- periodic training report ----
        if episode % REPORT_INTERVAL == 0:
            avg_len = statistics.mean(game_lengths[-REPORT_INTERVAL:])
            print_train_stats(
                episode=episode,
                stage=current_stage,
                total_wins=total_wins,
                total_losses=total_losses,
                total_draws=total_draws,
                rolling_wins=rolling_wins,
                rolling_total=len(rolling),
                rolling_wr=rolling_wr,
                avg_len=avg_len,
                epsilon=agent.epsilon,
                elapsed_s=time.perf_counter() - total_start,
            )

        # ---- greedy evaluation + promotion check ----
        if episode % EVAL_INTERVAL == 0 and current_stage < MAX_MCTS_DEPTH:
            eval_wins, eval_draws, eval_losses = run_eval(agent, opponent)
            promoted = eval_wins >= promote_threshold
            print_eval_stats(
                episode=episode,
                stage=current_stage,
                wins=eval_wins,
                draws=eval_draws,
                losses=eval_losses,
                threshold=promote_threshold,
                promoted=promoted,
            )

            if promoted:
                save_checkpoint(agent, episode, current_stage)
                current_stage += 1
                opponent = MCTSOpponent(depth=current_stage, n_simulations=MCTS_SIMULATIONS)
                # Reset counters so reported win-rate reflects the new stage only
                total_wins = total_losses = total_draws = 0
                rolling.clear()
                game_lengths.clear()
                print(f"  ╚═ Advanced to Stage {current_stage} (MCTS depth {current_stage})\n")

        # ---- periodic checkpoint ----
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