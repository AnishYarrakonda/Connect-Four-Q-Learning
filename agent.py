"""
agent.py — Deep Q-Network agent for 2048.

Architecture
────────────
Input encoding:  (16, 4, 4) one-hot tensor.
Network:         Conv2d(16→64) → Conv2d(64→128) → Conv2d(128→128) → FC(256) → FC(4)
Algorithm:       Double DQN, epsilon-greedy over valid actions, configurable reward shaping.
"""

from __future__ import annotations

import sys
import random
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import Board

DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DIRS       = ["up", "down", "left", "right"]
N_CHANNELS = 16


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  REWARD SHAPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class RewardConfig:
    """
    Every reward knob in one place. All controllable from the CLI via train.py.

    Recommended config for a fresh run:
        merge_weight=1.0, log_scale=True
        survival_bonus=0.0   ← no free points just for existing
        empty_weight=0.3     ← reward keeping board open (long-term)
        monotone_weight=0.2  ← reward snake tile ordering (long-term strategy)
        no_merge_penalty=0.5 ← punish wasted moves that slide but don't merge
        milestone_weight=2.0 ← big bonus for hitting a new max tile this game

    Knob guide
    ──────────
    merge_weight     — multiplier on raw merge score (naturally proportional:
                       merging two 1024s gives 512× more signal than two 2s)
    log_scale        — compress rewards with log1p so Q-values stay bounded
    survival_bonus   — flat reward per step. Set to 0 so agent must earn points.
    empty_weight     — bonus per free cell. Rewards keeping options open.
    monotone_weight  — bonus for sorted rows/cols (snake strategy). Try 0.1–0.3.
    no_merge_penalty — penalty when move slides tiles but merges nothing.
                       Discourages passive shuffling. Try 0.3–1.0.
    milestone_weight — bonus = weight × log2(new_max) when agent sets a new
                       personal-best tile this game. Rewards long-term building.
    """
    merge_weight:     float = 1.0
    log_scale:        bool  = True
    survival_bonus:   float = 0.0
    empty_weight:     float = 0.3
    monotone_weight:  float = 0.2
    no_merge_penalty: float = 0.5
    milestone_weight: float = 2.0


DEFAULT_REWARD_CFG = RewardConfig()


def _monotonicity(board_flat: np.ndarray) -> float:
    """Score how sorted the board is. Higher = more snake-ordered."""
    g = board_flat.reshape(4, 4).astype(np.float32)
    g = np.where(g > 0, np.log2(np.maximum(g, 1)), 0.0)
    score = 0.0
    for row in g:
        diff = np.diff(row)
        score += max(float(np.sum(diff[diff >= 0])), float(np.sum(-diff[diff <= 0])))
    for col in g.T:
        diff = np.diff(col)
        score += max(float(np.sum(diff[diff >= 0])), float(np.sum(-diff[diff <= 0])))
    return score / 48.0


def compute_reward(
    prev_score: int,
    prev_max:   int,
    board:      Board,
    cfg:        RewardConfig,
) -> float:
    merge_delta = float(board.score - prev_score)

    # merge reward — proportional by nature (bigger merges = bigger delta)
    reward = merge_delta * cfg.merge_weight

    # survival bonus (0.0 recommended — makes agent earn every point)
    reward += cfg.survival_bonus

    # empty cells bonus — rewards keeping the board open for future moves
    if cfg.empty_weight != 0.0:
        reward += cfg.empty_weight * float(np.sum(board.board == 0))

    # monotonicity bonus — rewards strategic tile ordering
    if cfg.monotone_weight != 0.0:
        reward += cfg.monotone_weight * _monotonicity(board.board)

    # no-merge penalty — punishes passive moves that waste a turn
    if cfg.no_merge_penalty != 0.0 and merge_delta == 0.0:
        reward -= cfg.no_merge_penalty

    # milestone bonus — big reward for setting a new personal-best tile this game
    if cfg.milestone_weight != 0.0:
        new_max = int(board.board.max())
        if new_max > prev_max:
            reward += cfg.milestone_weight * float(np.log2(new_max))

    if cfg.log_scale:
        reward = float(np.sign(reward) * np.log1p(abs(reward)))

    return reward


# ─────────────────────────────── encoding ─────────────────────────────────────
def encode(flat: np.ndarray) -> np.ndarray:
    out  = np.zeros((N_CHANNELS, 4, 4), dtype=np.float32)
    mask = flat > 0
    log2 = np.zeros(16, dtype=np.int64)
    log2[mask] = np.log2(flat[mask]).astype(np.int64)
    np.clip(log2, 0, N_CHANNELS - 1, out=log2)
    rows = np.arange(16) // 4
    cols = np.arange(16) % 4
    out[log2, rows, cols] = 1.0
    return out


# ─────────────────────────────── network ──────────────────────────────────────
class TwoZeroFourEightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(N_CHANNELS, 64,  kernel_size=2)
        self.conv2 = nn.Conv2d(64,         128, kernel_size=2)
        self.conv3 = nn.Conv2d(128,        128, kernel_size=2)
        self.fc1   = nn.Linear(128, 256)
        self.fc2   = nn.Linear(256, 4)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ─────────────────────────────── replay buffer ────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.cap  = capacity
        self.ptr  = 0
        self.size = 0
        self.states  = np.zeros((capacity, N_CHANNELS, 4, 4), dtype=np.float32)
        self.nstates = np.zeros((capacity, N_CHANNELS, 4, 4), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.bool_)

    def push(self, state: np.ndarray, action: int, reward: float,
             nstate: np.ndarray, done: bool) -> None:
        i = self.ptr
        self.states[i]  = state
        self.nstates[i] = nstate
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i]   = done
        self.ptr  = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, batch)
        return (
            torch.from_numpy(self.states[idx]).to(DEVICE),
            torch.from_numpy(self.actions[idx]).to(DEVICE),
            torch.from_numpy(self.rewards[idx]).to(DEVICE),
            torch.from_numpy(self.nstates[idx]).to(DEVICE),
            torch.from_numpy(self.dones[idx]).to(DEVICE),
        )

    def __len__(self) -> int:
        return self.size


# ─────────────────────────────── agent ────────────────────────────────────────
class DQNAgent:
    def __init__(
        self,
        lr:           float = 3e-4,
        gamma:        float = 0.99,
        eps_start:    float = 1.0,
        eps_end:      float = 0.02,
        eps_decay:    float = 0.9995,
        batch_size:   int   = 512,
        buffer_cap:   int   = 100_000,
        target_sync:  int   = 500,
        warmup:       int   = 2_000,
        learn_every:  int   = 4,
        reward_cfg:   RewardConfig | None = None,
    ):
        self.gamma       = gamma
        self.eps         = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.batch_size  = batch_size
        self.target_sync = target_sync
        self.warmup      = warmup
        self.learn_every = learn_every
        self.reward_cfg  = reward_cfg or DEFAULT_REWARD_CFG
        self.grad_steps  = 0
        self._step_count = 0
        # death penalty settings
        self.death_k    = 3.0   # scaling factor k (~2–5 suggested)
        self.death_back = 5     # how many moves before terminal to penalize

        self.policy = TwoZeroFourEightNet().to(DEVICE)
        self.target = TwoZeroFourEightNet().to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optim         = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer        = ReplayBuffer(buffer_cap)
        self.last_loss:     float = 0.0
        self.episode_count: int   = 0

    @torch.no_grad()
    def act(self, state_np: np.ndarray, valid: list[int]) -> int:
        if not valid:
            return 0
        if random.random() < self.eps:
            return random.choice(valid)
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)
        q = self.policy(state_t)[0]
        mask = torch.full((4,), float("-inf"), device=DEVICE)
        for a in valid:
            mask[a] = q[a]
        return int(mask.argmax())

    def learn(self) -> float | None:
        if len(self.buffer) < max(self.batch_size, self.warmup):
            return None
        s, a, r, ns, done = self.buffer.sample(self.batch_size)
        q_pred = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_next_a = self.policy(ns).argmax(1, keepdim=True)
            q_next      = self.target(ns).gather(1, best_next_a).squeeze(1)
            q_target    = r + self.gamma * q_next * (~done)
        loss = F.smooth_l1_loss(q_pred, q_target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optim.step()
        self.grad_steps += 1
        if self.grad_steps % self.target_sync == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.last_loss = float(loss.detach().item())
        return self.last_loss

    def run_episode(self, train: bool = True) -> tuple[int, int, int]:
        board    = Board()
        state    = encode(board.board)
        steps    = 0
        game_max = int(board.board.max())
        # track buffer indices for transitions pushed during this episode
        episode_push_indices: list[int] = []

        while not board.game_over:
            valid      = board.valid_actions()
            action     = self.act(state, valid) if train else self._greedy(state, valid)
            prev_score = board.score
            prev_max   = game_max
            board.move(DIRS[action])
            game_max   = max(game_max, int(board.board.max()))
            next_state = encode(board.board)

            if train:
                reward = compute_reward(prev_score, prev_max, board, self.reward_cfg)
                self.buffer.push(state, action, reward, next_state, board.game_over)
                # record index of pushed transition (push advanced ptr)
                last_idx = (self.buffer.ptr - 1) % self.buffer.cap
                episode_push_indices.append(int(last_idx))
                self._step_count += 1
                if self._step_count % self.learn_every == 0:
                    self.learn()

            state  = next_state
            steps += 1

        self.episode_count += 1

        # If episode ended in terminal (death) while training, apply retroactive
        # penalty distributed across the last `death_back` pushed transitions.
        if train and board.game_over and episode_push_indices:
            max_tile = int(board.board.max())
            if max_tile > 0:
                penalty_raw = self.death_k * math.log2(max_tile)
                # convert raw penalty to logged-domain penalty (negative)
                penalty_logged = -math.log1p(penalty_raw)
                N = min(self.death_back, len(episode_push_indices))
                weights = np.arange(N, 0, -1, dtype=np.float32)
                weights = weights / float(weights.sum())
                for i in range(N):
                    idx = episode_push_indices[-1 - i]
                    self.buffer.rewards[idx] = float(self.buffer.rewards[idx] + penalty_logged * weights[i])

        return board.score, int(board.board.max()), steps

    @torch.no_grad()
    def _greedy(self, state_np: np.ndarray, valid: list[int]) -> int:
        if not valid:
            return 0
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)
        q = self.policy(state_t)[0]
        mask = torch.full((4,), float("-inf"), device=DEVICE)
        for a in valid:
            mask[a] = q[a]
        return int(mask.argmax())

    def evaluate(self, n: int = 100) -> dict[str, object]:
        scores, tiles = [], []
        for _ in range(n):
            s, t, _ = self.run_episode(train=False)
            scores.append(s); tiles.append(t)
        tile_counts: dict[int, int] = {}
        for t in tiles:
            tile_counts[t] = tile_counts.get(t, 0) + 1
        return {
            "mean_score":  float(np.mean(scores)),
            "max_score":   int(max(scores)),
            "mean_tile":   float(np.mean(tiles)),
            "max_tile":    int(max(tiles)),
            "tile_counts": dict(sorted(tile_counts.items())),
            "n":           n,
        }

    def save(self, path: str = "agent.pt") -> None:
        torch.save({
            "policy":        self.policy.state_dict(),
            "target":        self.target.state_dict(),
            "optim":         self.optim.state_dict(),
            "eps":           self.eps,
            "grad_steps":    self.grad_steps,
            "episode_count": self.episode_count,
            "reward_cfg":    self.reward_cfg,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.optim.load_state_dict(ckpt["optim"])
        self.eps           = ckpt["eps"]
        self.grad_steps    = ckpt.get("grad_steps", 0)
        self.episode_count = ckpt.get("episode_count", 0)
        if "reward_cfg" in ckpt:
            self.reward_cfg = ckpt["reward_cfg"]


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    from train import train
    train(resume=ckpt)