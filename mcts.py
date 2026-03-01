"""
mcts.py — lightweight Monte Carlo opponent for curriculum training.

depth=0  → pure random (baseline)
depth=N  → N-ply random rollout to estimate move quality
n_simulations → number of rollouts per candidate move
"""

import random
from board import Board


class MCTSOpponent:
    def __init__(self, depth: int = 0, n_simulations: int = 40):
        """
        depth        : how many random plies to roll out after the candidate move.
                       0 = pick uniformly at random (no simulation).
        n_simulations: rollouts per legal move when depth > 0.
        """
        self.depth = depth
        self.n_simulations = n_simulations

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, board: Board) -> int:
        valid = board.valid_moves()
        if not valid:
            return 0

        # Depth-0 is pure random — no need to simulate.
        if self.depth == 0:
            return random.choice(valid)

        # First check for immediate wins or must-blocks.
        for col in valid:
            if self._is_winning_move(board, col):
                return col

        opponent_player = 2 if (board.turn % 2 == 0) else 1
        for col in valid:
            if self._is_winning_move_for(board, col, opponent_player):
                return col

        # Score remaining moves by simulation.
        scores = {col: 0.0 for col in valid}
        for col in valid:
            for _ in range(self.n_simulations):
                scores[col] += self._simulate(board, col)

        return max(valid, key=lambda c: scores[c])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clone(self, board: Board) -> Board:
        clone = Board()
        clone.player1_bits = board.player1_bits.clone()
        clone.player2_bits = board.player2_bits.clone()
        clone.turn = board.turn
        clone.move_history = board.move_history.copy()
        return clone

    def _is_winning_move(self, board: Board, col: int) -> bool:
        return self._is_winning_move_for(board, col, 1 if board.turn % 2 == 0 else 2)

    def _is_winning_move_for(self, board: Board, col: int, player_turn_parity: int) -> bool:
        """Check if playing col RIGHT NOW (given parity) results in a win."""
        sim = self._clone(board)
        row = sim.make_move(col)
        if row is None:
            return False
        done, winner = sim.game_over(row, col)
        return done and winner == player_turn_parity

    def _simulate(self, board: Board, first_col: int) -> float:
        """Return 1.0 win, 0.5 draw, 0.0 loss for the acting player."""
        sim = self._clone(board)
        acting_player = 1 if sim.turn % 2 == 0 else 2

        row = sim.make_move(first_col)
        if row is None:
            return 0.0

        done, winner = sim.game_over(row, first_col)
        if done:
            return _outcome(winner, acting_player)

        for _ in range(self.depth - 1):
            valid = sim.valid_moves()
            if not valid:
                break
            col = random.choice(valid)
            row = sim.make_move(col)
            if row is None:
                break
            done, winner = sim.game_over(row, col)
            if done:
                return _outcome(winner, acting_player)

        return 0.5  # no terminal reached → neutral


def _outcome(winner: int, acting_player: int) -> float:
    if winner == acting_player:
        return 1.0
    if winner == 0:
        return 0.5
    return 0.0