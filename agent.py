import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Optional

from board import Board, device

# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Two 3x3 convs with batch norm and a skip connection.
    Keeps spatial size fixed (same padding) — the full 6x7 board
    stays visible throughout, no spatial info is compressed away.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------
# Input:  (batch, 3, 6, 7)
#   channel 0 = current player's pieces
#   channel 1 = opponent's pieces
#   channel 2 = empty cells  ← encodes gravity / available moves explicitly
#
# Architecture:
#   Stem:  3x3 conv, same padding → 64 channels, board stays 6x7
#   Body:  4 residual blocks (3x3, same padding) — full 6x7 preserved
#   Head:  global avg pool → FC 128 → FC 7
#
# Why this is better than the old 4x4 kernel + shrinking design:
#   - 3x3 kernels are standard; 4x4 with padding=1 misaligns with 4-in-a-row patterns
#   - same padding keeps the board 6x7 through all layers — no spatial info lost
#   - residual connections give gradients a direct path back, stable training
#   - 64 channels (not 128) is right-sized for 6x7; fewer params, trains faster
#   - global avg pool before FC gives position-invariant feature aggregation
#   - 3rd empty-channel teaches the net which columns are playable and where
#     gravity means pieces will land
# ---------------------------------------------------------------------------

class ConnectFourNet(nn.Module):
    CHANNELS = 64

    def __init__(self):
        super().__init__()

        # Stem: 3 input channels → 64, keep full 6x7
        self.stem    = nn.Conv2d(3, self.CHANNELS, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(self.CHANNELS)

        # Body: 4 residual blocks, all 6x7
        self.res_blocks = nn.Sequential(
            ResBlock(self.CHANNELS),
            ResBlock(self.CHANNELS),
            ResBlock(self.CHANNELS),
            ResBlock(self.CHANNELS),
        )

        # Head: global avg pool → 64 → 128 → 7
        self.fc1 = nn.Linear(self.CHANNELS, 128)
        self.fc2 = nn.Linear(128, Board.COLS)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 3, 6, 7)
        returns: (batch, 7) raw logits
        """
        x = F.relu(self.stem_bn(self.stem(x)))  # → (b, 64, 6, 7)
        x = self.res_blocks(x)                   # → (b, 64, 6, 7)
        x = F.adaptive_avg_pool2d(x, 1)          # → (b, 64, 1, 1)
        x = x.flatten(start_dim=1)               # → (b, 64)
        x = F.relu(self.fc1(x))                  # → (b, 128)
        return self.fc2(x)                        # → (b, 7)

    def policy(self, board: Board) -> torch.Tensor:
        """Masked softmax over valid columns. Returns (7,) tensor."""
        self.eval()
        with torch.no_grad():
            state  = Board.board_to_tensor(board).view(1, 3, Board.ROWS, Board.COLS)
            logits = self(state).squeeze(0)

            mask = torch.full((Board.COLS,), float('-inf'), device=device)
            for col in board.valid_moves():
                mask[col] = 0.0

            probs = F.softmax(logits + mask, dim=0)
        return probs

    def best_move(self, board: Board) -> int:
        return int(self.policy(board).argmax().item())

    def sample_move(self, board: Board, temperature: float = 1.0) -> int:
        probs = self.policy(board)
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1).item())


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states).to(device),
            torch.tensor(actions,  dtype=torch.long,    device=device),
            torch.tensor(rewards,  dtype=torch.float32, device=device),
            torch.cat(next_states).to(device),
            torch.tensor(dones,    dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        target_update_freq: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        buffer_capacity: int = 50_000,
    ):
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start      = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.steps_done         = 0

        self.policy_net = ConnectFourNet().to(device)
        self.target_net = ConnectFourNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

    @property
    def epsilon(self) -> float:
        return float(
            self.epsilon_end + (self.epsilon_start - self.epsilon_end)
            * np.exp(-self.steps_done / self.epsilon_decay)
        )

    def select_action(self, board: Board) -> int:
        valid = board.valid_moves()
        if random.random() < self.epsilon:
            return random.choice(valid)
        return self.policy_net.best_move(board)

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Reshape to (b, 3, 6, 7) — 3 channels now
        s  = states.view(-1, 3, Board.ROWS, Board.COLS)
        ns = next_states.view(-1, 3, Board.ROWS, Board.COLS)

        self.policy_net.train()
        q_values = self.policy_net(s).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str = "agent.pth"):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)
        print(f"[Agent] Saved → {path}")

    def load(self, path: str = "agent.pth"):
        ckpt = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt.get("steps_done", 0)
        print(f"[Agent] Loaded ← {path}  (step {self.steps_done})")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    board = Board()
    agent = DQNAgent()

    print(f"Device: {device}")
    print(f"Params: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

    dummy = Board.board_to_tensor(board).view(1, 3, Board.ROWS, Board.COLS)
    out   = agent.policy_net(dummy)
    print(f"Output shape: {out.shape}")   # torch.Size([1, 7])

    action = agent.select_action(board)
    print(f"Selected action: {action}")
    print("Sanity check passed ✓")