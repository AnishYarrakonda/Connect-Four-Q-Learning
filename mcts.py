"""
mcts.py — fast Monte Carlo opponent for curriculum training.

The key optimisation: all simulation uses a pure-Python FastBoard (flat int
list, no PyTorch) so cloning and move-making inside rollouts costs nothing.
The public interface still accepts the tensor-based Board from board.py.

depth=0  → pure random (no simulation)
depth=N  → N-ply random rollouts to score each candidate move
"""

import random
from board import Board

ROWS = 6
COLS = 7

# Precompute all winning lines as flat index tuples once at import time.
# Each entry is a tuple of 4 board indices (row*COLS + col).
def _build_win_lines() -> list[tuple[int, ...]]:
    lines = []
    for r in range(ROWS):
        for c in range(COLS):
            # horizontal
            if c + 3 < COLS:
                lines.append((r*COLS+c, r*COLS+c+1, r*COLS+c+2, r*COLS+c+3))
            # vertical
            if r + 3 < ROWS:
                lines.append((r*COLS+c, (r+1)*COLS+c, (r+2)*COLS+c, (r+3)*COLS+c))
            # diagonal \
            if r + 3 < ROWS and c + 3 < COLS:
                lines.append((r*COLS+c, (r+1)*COLS+c+1, (r+2)*COLS+c+2, (r+3)*COLS+c+3))
            # diagonal /
            if r + 3 < ROWS and c - 3 >= 0:
                lines.append((r*COLS+c, (r+1)*COLS+c-1, (r+2)*COLS+c-2, (r+3)*COLS+c-3))
    return lines

_WIN_LINES = _build_win_lines()


# ---------------------------------------------------------------------------
# FastBoard — pure Python, no PyTorch, designed for cheap cloning
# ---------------------------------------------------------------------------

class FastBoard:
    """
    Minimal board using a flat int list: 0=empty, 1=P1, 2=P2.
    column_height[c] tracks the next empty row in column c.
    No PyTorch anywhere — clone() is just list.copy().
    """
    __slots__ = ("cells", "heights", "turn")

    def __init__(self) -> None:
        self.cells:   list[int] = [0] * (ROWS * COLS)
        self.heights: list[int] = [0] * COLS
        self.turn: int = 0          # 0-indexed; player = (turn%2)+1

    # ---- construction from a tensor Board ----

    @staticmethod
    def from_board(board: Board) -> "FastBoard":
        fb = FastBoard()
        for r in range(ROWS):
            for c in range(COLS):
                if board.player1_bits[r, c] == 1:
                    fb.cells[r * COLS + c] = 1
                elif board.player2_bits[r, c] == 1:
                    fb.cells[r * COLS + c] = 2
        # Recompute column heights
        for c in range(COLS):
            h = 0
            for r in range(ROWS):
                if fb.cells[r * COLS + c] != 0:
                    h = r + 1
            fb.heights[c] = h
        fb.turn = board.turn
        return fb

    def clone(self) -> "FastBoard":
        fb = FastBoard()
        fb.cells   = self.cells.copy()
        fb.heights = self.heights.copy()
        fb.turn    = self.turn
        return fb

    def valid_moves(self) -> list[int]:
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    def make_move(self, col: int) -> int:
        """Drop a piece; returns the row placed, or -1 if column full."""
        r = self.heights[col]
        if r >= ROWS:
            return -1
        player = (self.turn % 2) + 1
        self.cells[r * COLS + col] = player
        self.heights[col] += 1
        self.turn += 1
        return r

    def last_player(self) -> int:
        """The player who just moved."""
        return ((self.turn - 1) % 2) + 1

    def check_win(self, last_player: int) -> bool:
        c = last_player
        cells = self.cells
        for a, b, d, e in _WIN_LINES:
            if cells[a] == c and cells[b] == c and cells[d] == c and cells[e] == c:
                return True
        return False

    def is_full(self) -> bool:
        return self.turn >= ROWS * COLS


# ---------------------------------------------------------------------------
# MCTSOpponent
# ---------------------------------------------------------------------------

class MCTSOpponent:
    def __init__(self, depth: int = 0, n_simulations: int = 20):
        """
        depth        : plies to roll out per simulation (0 = pure random pick).
        n_simulations: rollouts per candidate move when depth > 0.
                       Default lowered to 20 — plenty of signal, much faster.
        """
        self.depth        = depth
        self.n_simulations = n_simulations

    # ------------------------------------------------------------------
    # Public interface  (accepts the tensor Board from board.py)
    # ------------------------------------------------------------------

    def select_action(self, board: Board) -> int:
        fb    = FastBoard.from_board(board)
        valid = fb.valid_moves()
        if not valid:
            return 0

        if self.depth == 0:
            return random.choice(valid)

        acting = (fb.turn % 2) + 1       # player about to move
        opp    = 3 - acting

        # Immediate win
        for col in valid:
            if self._wins_immediately(fb, col, acting):
                return col

        # Must-block opponent win
        for col in valid:
            if self._wins_immediately(fb, col, opp):
                return col

        # Score by rollout
        scores = [0.0] * COLS
        for col in valid:
            for _ in range(self.n_simulations):
                scores[col] += self._simulate(fb, col, acting)

        return max(valid, key=lambda c: scores[c])

    # ------------------------------------------------------------------
    # Internal helpers — all work on FastBoard
    # ------------------------------------------------------------------

    def _wins_immediately(self, fb: FastBoard, col: int, player: int) -> bool:
        sim = fb.clone()
        r   = sim.make_move(col)
        if r < 0:
            return False
        return sim.check_win(player)

    def _simulate(self, fb: FastBoard, first_col: int, acting: int) -> float:
        sim = fb.clone()
        r   = sim.make_move(first_col)
        if r < 0:
            return 0.0
        if sim.check_win(acting):
            return 1.0
        if sim.is_full():
            return 0.5

        for _ in range(self.depth - 1):
            valid = sim.valid_moves()
            if not valid:
                return 0.5
            col = random.choice(valid)
            r   = sim.make_move(col)
            last = sim.last_player()
            if sim.check_win(last):
                return 1.0 if last == acting else 0.0
            if sim.is_full():
                return 0.5

        return 0.5