# imports
import random
from typing import Optional

import torch
import torch.nn as nn

from board import Board


class ConnectFourCNN(nn.Module):
    """CNN Q-network consuming flattened (2*6*7 + 1) input."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        conv_out = 128 * Board.ROWS * Board.COLS
        self.head = nn.Sequential(
            nn.Linear(conv_out + 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, Board.COLS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x[:, : 2 * Board.ROWS * Board.COLS]
        turn = x[:, -1:]
        spatial = spatial.view(-1, 2, Board.ROWS, Board.COLS)
        features = self.conv(spatial).flatten(1)
        combined = torch.cat([features, turn], dim=1)
        return self.head(combined)


class Agent:
    def __init__(
        self: "Agent",
        layers: list[int],  # kept for compatibility with existing config flow
        lr: float = 0.0003,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.997,
        epsilon_min: float = 0.05,
        gamma: float = 0.97,
    ) -> None:
        _ = layers
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model: nn.Module = ConnectFourCNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

    def predict(self: "Agent", board: Board) -> torch.Tensor:
        state = Board.board_to_tensor(board).to(self.device)
        with torch.no_grad():
            return self.model(state)

    def select_action(self: "Agent", board: Board, valid_moves: Optional[list[int]] = None) -> int:
        if valid_moves is None:
            valid_moves = board.valid_moves()
        if not valid_moves:
            return 0
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        state = Board.board_to_tensor(board).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)[0]
        masked = q_values.clone()
        for col in range(Board.COLS):
            if col not in valid_moves:
                masked[col] = -float("inf")
        return int(torch.argmax(masked).item())

    def train_step(
        self: "Agent",
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)

        self.model.train()
        q_values = self.model(state)
        self.model.eval()
        with torch.no_grad():
            next_q = self.model(next_state)

        target_q = q_values.clone().detach()
        target_value = reward if done else reward + self.gamma * torch.max(next_q).item()
        target_q[0, action] = torch.tensor(target_value, dtype=torch.float32, device=self.device)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
