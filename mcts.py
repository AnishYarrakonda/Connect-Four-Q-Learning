"""
mcts.py — maximally optimised Monte Carlo opponent.

Key optimisations vs previous version:
  1. check_win only scans lines through the last-placed cell (not all 69)
  2. from_board uses .tolist() — one tensor op instead of 42 element accesses
  3. _wins_immediately never clones — checks inline with undo
  4. valid_moves inlined as a tuple in hot paths
  5. simulate uses local variable aliases to avoid attribute lookups
  6. FastBoard stores cells as a bytearray (faster copy than list)
  7. Per-column win-line index precomputed at import time
"""

import random
from board import Board

ROWS = 6
COLS = 7
SIZE = ROWS * COLS

# ---------------------------------------------------------------------------
# Precompute win lines, indexed two ways:
#   _WIN_LINES       — all 69 lines (list of 4-tuples of flat indices)
#   _CELL_WIN_LINES  — for each cell, which lines pass through it
# ---------------------------------------------------------------------------

def _build_indices():
    lines = []
    for r in range(ROWS):
        for c in range(COLS):
            if c + 3 < COLS:
                lines.append((r*COLS+c, r*COLS+c+1, r*COLS+c+2, r*COLS+c+3))
            if r + 3 < ROWS:
                lines.append((r*COLS+c, (r+1)*COLS+c, (r+2)*COLS+c, (r+3)*COLS+c))
            if r + 3 < ROWS and c + 3 < COLS:
                lines.append((r*COLS+c, (r+1)*COLS+c+1, (r+2)*COLS+c+2, (r+3)*COLS+c+3))
            if r + 3 < ROWS and c - 3 >= 0:
                lines.append((r*COLS+c, (r+1)*COLS+c-1, (r+2)*COLS+c-2, (r+3)*COLS+c-3))

    # For each cell, precompute which lines pass through it
    cell_lines = [[] for _ in range(SIZE)]
    for line in lines:
        for idx in line:
            cell_lines[idx].append(line)

    return lines, [tuple(cl) for cl in cell_lines]

_WIN_LINES, _CELL_WIN_LINES = _build_indices()


# ---------------------------------------------------------------------------
# FastBoard — bytearray cells, minimal overhead
# ---------------------------------------------------------------------------

class FastBoard:
    __slots__ = ("cells", "heights", "turn")

    def __init__(self):
        self.cells   = bytearray(SIZE)   # bytearray.copy() is faster than list.copy()
        self.heights = [0] * COLS
        self.turn    = 0

    @staticmethod
    def from_board(board: Board) -> "FastBoard":
        fb = FastBoard()
        # One .tolist() call per player instead of 42 individual tensor accesses
        p1 = board.player1_bits.tolist()
        p2 = board.player2_bits.tolist()
        cells = fb.cells
        heights = fb.heights
        for r in range(ROWS):
            for c in range(COLS):
                if p1[r][c]:
                    idx = r * COLS + c
                    cells[idx] = 1
                    if r + 1 > heights[c]:
                        heights[c] = r + 1
                elif p2[r][c]:
                    idx = r * COLS + c
                    cells[idx] = 2
                    if r + 1 > heights[c]:
                        heights[c] = r + 1
        fb.turn = board.turn
        return fb

    def clone(self) -> "FastBoard":
        fb = FastBoard()
        fb.cells   = bytearray(self.cells)   # fast C-level copy
        fb.heights = self.heights.copy()
        fb.turn    = self.turn
        return fb

    def make_move(self, col: int) -> int:
        r = self.heights[col]
        if r >= ROWS:
            return -1
        self.cells[r * COLS + col] = (self.turn & 1) + 1
        self.heights[col] = r + 1
        self.turn += 1
        return r

    def check_win_at(self, cell: int, player: int) -> bool:
        """Only scan win lines that pass through `cell` — much faster than full scan."""
        c = player
        cells = self.cells
        for a, b, d, e in _CELL_WIN_LINES[cell]:
            if cells[a] == c and cells[b] == c and cells[d] == c and cells[e] == c:
                return True
        return False

    def is_full(self) -> bool:
        return self.turn >= SIZE


# ---------------------------------------------------------------------------
# MCTSOpponent
# ---------------------------------------------------------------------------

class MCTSOpponent:
    def __init__(self, depth: int = 0, n_simulations: int = 20):
        self.depth         = depth
        self.n_simulations = n_simulations

    def select_action(self, board: Board) -> int:
        return self._pick(FastBoard.from_board(board))

    def select_action_fast(self, fb: FastBoard) -> int:
        return self._pick(fb)

    def _pick(self, fb: FastBoard) -> int:
        heights = fb.heights
        valid   = [c for c in range(COLS) if heights[c] < ROWS]
        if not valid:
            return 0

        if self.depth == 0:
            return random.choice(valid)

        acting = (fb.turn & 1) + 1
        opp    = 3 - acting

        # Immediate win — check inline with undo (no clone)
        cells = fb.cells
        for col in valid:
            r   = heights[col]
            idx = r * COLS + col
            cells[idx] = acting
            win = fb.check_win_at(idx, acting)
            cells[idx] = 0
            if win:
                return col

        # Must-block
        for col in valid:
            r   = heights[col]
            idx = r * COLS + col
            cells[idx] = opp
            win = fb.check_win_at(idx, opp)
            cells[idx] = 0
            if win:
                return col

        # Rollout scoring
        scores  = [0.0] * COLS
        n_sims  = self.n_simulations
        depth   = self.depth
        simulate = self._simulate
        for col in valid:
            s = 0.0
            for _ in range(n_sims):
                s += simulate(fb, col, acting, depth)
            scores[col] = s

        return max(valid, key=lambda c: scores[c])

    def _simulate(self, fb: FastBoard, first_col: int, acting: int, depth: int) -> float:
        sim = fb.clone()
        r   = sim.make_move(first_col)
        if r < 0:
            return 0.0

        idx = r * COLS + first_col
        if sim.check_win_at(idx, acting):
            return 1.0
        if sim.is_full():
            return 0.5

        heights = sim.heights
        cells   = sim.cells
        turn    = sim.turn

        for _ in range(depth - 1):
            valid = [c for c in range(COLS) if heights[c] < ROWS]
            if not valid:
                return 0.5
            col  = random.choice(valid)
            r    = heights[col]
            idx  = r * COLS + col
            last = (turn & 1) + 1
            cells[idx]   = last
            heights[col] = r + 1
            turn        += 1
            sim.turn     = turn

            if sim.check_win_at(idx, last):
                return 1.0 if last == acting else 0.0
            if turn >= SIZE:
                return 0.5

        return 0.5