"""
gui.py — Connect Four GUI adapted for the tensor-based Board and DQNAgent.

Controls:
  Load P1 / P2 Model  — load a .pth checkpoint for either side
  Mode                — Human vs AI / Human vs Human / Agent vs Agent
  Human Side          — which player the human controls in HvA mode
  Start / Reset       — start or restart the current game
  AI Speed slider     — delay between AI moves (ms)
"""

import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import torch

from board import Board, device
from agent import DQNAgent, ConnectFourNet


# ---------------------------------------------------------------------------
# Thin AI wrapper (keeps the GUI decoupled from DQN internals)
# ---------------------------------------------------------------------------

class GUIAgent:
    def __init__(self) -> None:
        self.net: Optional[ConnectFourNet] = None
        self.model_path: Optional[str] = None

    def load(self, path: str) -> tuple[bool, str]:
        try:
            ckpt = torch.load(path, map_location=device)
            net = ConnectFourNet().to(device)

            # Accept both a raw state_dict and a DQNAgent checkpoint
            if isinstance(ckpt, dict) and "policy_net" in ckpt:
                net.load_state_dict(ckpt["policy_net"])
            elif isinstance(ckpt, dict):
                net.load_state_dict(ckpt)
            else:
                raise ValueError("Unrecognised checkpoint format.")

            net.eval()
            self.net = net
            self.model_path = path
            return True, f"Loaded: {os.path.basename(path)}"
        except Exception as exc:
            self.net = None
            self.model_path = None
            return False, f"Failed to load model: {exc}"

    def select_move(self, board: Board) -> int:
        valid = board.valid_moves()
        if not valid:
            return 0
        if self.net is None:
            return random.choice(valid)
        return self.net.best_move(board)

    @property
    def display_name(self) -> str:
        return os.path.basename(self.model_path) if self.model_path else "random-policy"


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class ConnectFourGUI:
    CELL          = 92
    PADDING       = 12
    BOARD_COLOR   = "#1769aa"
    EMPTY_COLOR   = "#f2f7fb"
    P1_COLOR      = "#d62828"
    P2_COLOR      = "#f4c430"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Connect Four")
        self.root.resizable(False, False)

        self.board = Board()
        self.agent_p1 = GUIAgent()
        self.agent_p2 = GUIAgent()

        # --- state vars ---
        self.mode_var        = tk.StringVar(value="Human vs AI")
        self.human_side_var  = tk.StringVar(value="First (Player 1)")
        self.turn_var        = tk.StringVar(value="Press Start to begin")
        self.result_var      = tk.StringVar(value="")
        self.match_var       = tk.StringVar(value="")
        self.model_p1_var    = tk.StringVar(value="P1: random-policy")
        self.model_p2_var    = tk.StringVar(value="P2: random-policy")

        self.game_over       = False
        self.game_started    = False
        self.animating       = False
        self.ai_job: Optional[str]           = None
        self.active_anim_job: Optional[str]  = None
        self.active_token_id: Optional[int]  = None
        self.anim_gen        = 0

        self.token_ids: list[list[Optional[int]]] = [
            [None] * Board.COLS for _ in range(Board.ROWS)
        ]

        self._build_controls()
        self._build_canvas()
        self._redraw_board()
        self._update_match_label()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=6)

        tk.Label(bar, text="Mode:").grid(row=0, column=0, sticky="w", padx=(0, 3))
        tk.OptionMenu(
            bar, self.mode_var,
            "Human vs AI", "Human vs Human", "Agent vs Agent",
            command=lambda _: self._on_mode_change(),
        ).grid(row=0, column=1, sticky="w")

        tk.Label(bar, text="Side:").grid(row=0, column=2, sticky="w", padx=(10, 3))
        tk.OptionMenu(
            bar, self.human_side_var,
            "First (Player 1)", "Second (Player 2)",
            command=lambda _: self._on_mode_change(),
        ).grid(row=0, column=3, sticky="w")

        tk.Button(bar, text="Load P1 Model", command=lambda: self._load_model(1)).grid(row=0, column=4, padx=6)
        tk.Button(bar, text="Load P2 Model", command=lambda: self._load_model(2)).grid(row=0, column=5, padx=6)
        tk.Button(bar, text="Start",  command=self.start_game).grid(row=0, column=6, padx=4)
        tk.Button(bar, text="Reset",  command=self.reset_game).grid(row=0, column=7, padx=4)
        tk.Button(bar, text="Quit",   command=self.root.destroy).grid(row=0, column=8, padx=4)

        tk.Label(bar, text="AI ms:").grid(row=0, column=9, padx=(10, 3))
        self.ai_speed = tk.Scale(bar, from_=50, to=1000, orient="horizontal", length=140)
        self.ai_speed.set(300)
        self.ai_speed.grid(row=0, column=10, sticky="w")

        for var, font in [
            (self.match_var,   ("Helvetica", 10)),
            (self.turn_var,    ("Helvetica", 12, "bold")),
            (self.result_var,  ("Helvetica", 11)),
            (self.model_p1_var,("Helvetica", 10)),
            (self.model_p2_var,("Helvetica", 10)),
        ]:
            tk.Label(self.root, textvariable=var, font=font).pack(anchor="w", padx=10)

    def _build_canvas(self) -> None:
        w = Board.COLS * self.CELL
        h = Board.ROWS * self.CELL
        self.canvas = tk.Canvas(self.root, width=w, height=h, bg="#0e3d63", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._on_click)

    # ------------------------------------------------------------------
    # Board rendering
    # ------------------------------------------------------------------

    def _cell_bbox(self, row: int, col: int) -> tuple[float, float, float, float]:
        vr = Board.ROWS - 1 - row
        x0 = col * self.CELL + self.PADDING
        y0 = vr  * self.CELL + self.PADDING
        x1 = (col + 1) * self.CELL - self.PADDING
        y1 = (vr  + 1) * self.CELL - self.PADDING
        return x0, y0, x1, y1

    def _redraw_board(self) -> None:
        self.canvas.delete("all")
        w = Board.COLS * self.CELL
        h = Board.ROWS * self.CELL
        self.canvas.create_rectangle(0, 0, w, h, fill=self.BOARD_COLOR, outline=self.BOARD_COLOR)

        for r in range(Board.ROWS):
            for c in range(Board.COLS):
                x0, y0, x1, y1 = self._cell_bbox(r, c)
                self.canvas.create_oval(x0, y0, x1, y1, fill=self.EMPTY_COLOR, outline="#d9e9f7", width=2)

        # Re-draw any existing tokens (e.g. after a reset that kept board state)
        for r in range(Board.ROWS):
            for c in range(Board.COLS):
                if self.board.player1_bits[r, c] == 1:
                    self.token_ids[r][c] = self._draw_token(r, c, 1)
                elif self.board.player2_bits[r, c] == 1:
                    self.token_ids[r][c] = self._draw_token(r, c, 2)

    def _draw_token(self, row: int, col: int, player: int) -> int:
        color = self.P1_COLOR if player == 1 else self.P2_COLOR
        x0, y0, x1, y1 = self._cell_bbox(row, col)
        return self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline="#222", width=2)

    # ------------------------------------------------------------------
    # Game flow
    # ------------------------------------------------------------------

    def _player_label(self, player: int) -> str:
        return "Red" if player == 1 else "Yellow"

    @property
    def _current_player(self) -> int:
        return 1 if self.board.turn % 2 == 0 else 2

    def start_game(self) -> None:
        self._cancel_ai()
        self._cancel_anim()
        self.game_over    = False
        self.game_started = True
        self.result_var.set("")
        p = self._current_player
        self.turn_var.set(f"Turn: Player {p} ({self._player_label(p)})")
        self._schedule_ai()

    def reset_game(self) -> None:
        self._cancel_ai()
        self._cancel_anim()
        self.board.reset()
        self.game_over    = False
        self.game_started = False
        self.animating    = False
        self.result_var.set("")
        self.turn_var.set("Press Start to begin")
        self.token_ids = [[None] * Board.COLS for _ in range(Board.ROWS)]
        self._redraw_board()
        self._update_match_label()

    def _on_mode_change(self) -> None:
        self._update_match_label()
        self.reset_game()

    def _load_model(self, player: int) -> None:
        path = filedialog.askopenfilename(
            title="Load checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if not path:
            return
        agent = self.agent_p1 if player == 1 else self.agent_p2
        ok, msg = agent.load(path)
        label_var = self.model_p1_var if player == 1 else self.model_p2_var
        label_var.set(f"P{player}: {agent.display_name}")
        self._update_match_label()
        (messagebox.showinfo if ok else messagebox.showerror)("Model", msg)

    def _update_match_label(self) -> None:
        mode = self.mode_var.get()
        if mode == "Human vs Human":
            text = "Player 1: Human  |  Player 2: Human"
        elif mode == "Agent vs Agent":
            text = (f"Player 1: CPU ({self.agent_p1.display_name})  |  "
                    f"Player 2: CPU ({self.agent_p2.display_name})")
        elif self.human_side_var.get().startswith("Second"):
            text = (f"Player 1: CPU ({self.agent_p1.display_name})  |  Player 2: Human")
        else:
            text = (f"Player 1: Human  |  Player 2: CPU ({self.agent_p2.display_name})")
        self.match_var.set(text)

    # ------------------------------------------------------------------
    # Human / AI routing
    # ------------------------------------------------------------------

    def _is_human_turn(self) -> bool:
        mode = self.mode_var.get()
        if mode == "Human vs Human":
            return True
        if mode == "Agent vs Agent":
            return False
        # Human vs AI
        second = self.human_side_var.get().startswith("Second")
        return self._current_player == (2 if second else 1)

    def _is_ai_turn(self) -> bool:
        return not self._is_human_turn()

    def _ai_agent(self) -> GUIAgent:
        return self.agent_p1 if self._current_player == 1 else self.agent_p2

    def _schedule_ai(self) -> None:
        if self.game_over or self.animating or not self.game_started:
            return
        if not self._is_ai_turn():
            return
        self.ai_job = self.root.after(int(self.ai_speed.get()), self._run_ai)

    def _run_ai(self) -> None:
        self.ai_job = None
        if self.game_over or self.animating or not self.game_started:
            return
        col = self._ai_agent().select_move(self.board)
        self._try_move(col)

    def _cancel_ai(self) -> None:
        if self.ai_job:
            self.root.after_cancel(self.ai_job)
            self.ai_job = None

    # ------------------------------------------------------------------
    # Move execution
    # ------------------------------------------------------------------

    def _on_click(self, event: tk.Event) -> None:
        if self.game_over or self.animating or not self.game_started:
            return
        if not self._is_human_turn():
            return
        col = max(0, min(Board.COLS - 1, event.x // self.CELL))
        self._try_move(col)

    def _try_move(self, col: int) -> None:
        if self.animating or self.game_over or not self.game_started:
            return
        if self.board.is_column_full(col):
            self.result_var.set(f"Column {col + 1} is full — pick another.")
            return
        self.result_var.set("")

        player = self._current_player
        row = self.board.make_move(col)
        if row is None:
            return

        self.animating = True
        self._animate(col, row, player, callback=lambda: self._finalize(row, col, player))

    def _finalize(self, row: int, col: int, player: int) -> None:
        self.animating = False
        done, winner = self.board.game_over(row, col)

        if done:
            self.game_over = True
            self.turn_var.set("Game Over")
            if winner == 0:
                self.result_var.set("Draw!")
            else:
                self.result_var.set(f"Player {winner} ({self._player_label(winner)}) wins!")
            return

        p = self._current_player
        self.turn_var.set(f"Turn: Player {p} ({self._player_label(p)})")
        self._schedule_ai()

    # ------------------------------------------------------------------
    # Drop animation
    # ------------------------------------------------------------------

    def _cancel_anim(self) -> None:
        self.anim_gen += 1
        if self.active_anim_job:
            self.root.after_cancel(self.active_anim_job)
            self.active_anim_job = None
        if self.active_token_id is not None:
            self.canvas.delete(self.active_token_id)
            self.active_token_id = None
        self.animating = False

    def _animate(self, col: int, row: int, player: int, callback) -> None:
        gen = self.anim_gen
        color = self.P1_COLOR if player == 1 else self.P2_COLOR
        x0, y0, x1, y1 = self._cell_bbox(row, col)
        diam = x1 - x0
        cx   = (x0 + x1) / 2
        r    = diam / 2
        target_y = (y0 + y1) / 2

        tid = self.canvas.create_oval(
            cx - r, -r - 10, cx + r, r - 10,
            fill=color, outline="#222", width=2,
        )
        self.active_token_id = tid

        def step(cy: float, vy: float, bounces: int) -> None:
            if gen != self.anim_gen:
                self.canvas.delete(tid)
                return
            vy += 2.2
            cy += vy
            if cy >= target_y:
                cy = target_y
                if bounces < 1:
                    vy = -max(2.0, vy * 0.35)
                    bounces += 1
                else:
                    self.canvas.delete(tid)
                    self.active_token_id = None
                    self.active_anim_job = None
                    self.token_ids[row][col] = self._draw_token(row, col, player)
                    callback()
                    return
            self.canvas.coords(tid, cx - r, cy - r, cx + r, cy + r)
            self.active_anim_job = self.root.after(16, lambda: step(cy, vy, bounces))

        step(-r - 10, 0.0, 0)


def main() -> None:
    root = tk.Tk()
    ConnectFourGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
