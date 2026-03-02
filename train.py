"""
train.py — Training loop for the 2048 DQN agent.

Usage examples
──────────────
  # Fresh run with recommended settings:
  python train.py --lr 3e-4 --learn-every 4 --eps-decay 0.9995 --eps-end 0.02

  # Resume a checkpoint:
  python train.py -r models/ckpt_ep003000.pt --lr 3e-4 --learn-every 4

  # Aggressive reward shaping:
  python train.py --empty-weight 0.3 --monotone-weight 0.2 --no-merge-penalty 0.5

  # See all flags:
  python train.py --help

"""

from __future__ import annotations

import argparse
import os
import re
import signal
import time
from pathlib import Path

import numpy as np

from agent import DQNAgent, TwoZeroFourEightNet, RewardConfig, DEFAULT_REWARD_CFG, DEVICE


# ─────────────────────────────── ANSI helpers ─────────────────────────────────
_ESC = "\033["
def _c(code: str, text: str) -> str: return f"{_ESC}{code}m{text}{_ESC}0m"

def dim(t: str) -> str:      return _c("2",    t)
def green(t: str) -> str:    return _c("32",   t)
def yellow(t: str) -> str:   return _c("33",   t)
def magenta(t: str) -> str:  return _c("35",   t)
def cyan(t: str) -> str:     return _c("36",   t)
def white(t: str) -> str:    return _c("37",   t)
def bred(t: str) -> str:     return _c("1;31", t)
def bgreen(t: str) -> str:   return _c("1;32", t)
def byellow(t: str) -> str:  return _c("1;33", t)
def bblue(t: str) -> str:    return _c("1;34", t)
def bmagenta(t: str) -> str: return _c("1;35", t)
def bcyan(t: str) -> str:    return _c("1;36", t)
def bwhite(t: str) -> str:   return _c("1;37", t)

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
def vis(s: str) -> int:
    return len(_ANSI_RE.sub('', s))


# ─────────────────────────────── layout ───────────────────────────────────────
W_EP = 15; W_ASC = 9; W_MSC = 9; W_MV = 7; W_TILE = 11; W_EPS = 7; W_BUF = 7; W_LOSS = 8
_SEP    = dim(" │ ")
TABLE_W = W_EP + W_ASC + W_MSC + W_MV + W_TILE + W_EPS + W_BUF + W_LOSS + 7 * 3  # 94
BOX_IN  = TABLE_W - 4  # 90

def _box_row(content: str) -> str:
    return bwhite("  ║") + content + " " * max(0, BOX_IN - vis(content)) + bwhite("║")
def _box_blank() -> str:
    return bwhite("  ║") + " " * BOX_IN + bwhite("║")
def _hcol(label: str, w: int, fn: "function") -> str:  # type: ignore[type-arg]
    return fn(label.center(w))# type: ignore

HEADER = (
    _hcol("EPISODE",  W_EP,   bcyan)   + _SEP# type: ignore
  + _hcol("AVG SCORE",W_ASC,  byellow) + _SEP# type: ignore
  + _hcol("MAX SCORE",W_MSC,  bwhite)  + _SEP# type: ignore
  + _hcol("MOVES",    W_MV,   bblue)   + _SEP# type: ignore
  + _hcol("MED  MAX", W_TILE, byellow) + _SEP# type: ignore
  + _hcol("ε",        W_EPS,  bmagenta)+ _SEP# type: ignore
  + _hcol("BUF",      W_BUF,  dim)     + _SEP# type: ignore
  + _hcol("LOSS",     W_LOSS, dim)# type: ignore
)
DIVIDER = dim("─" * TABLE_W)


# ─────────────────────────────── coloring ─────────────────────────────────────
def _tile_color(tile: int, w: int = 5) -> str:
    s = str(tile).rjust(w)
    if tile >= 2048: return bred(s)
    if tile >= 1024: return byellow(s)
    if tile >=  512: return yellow(s)
    if tile >=  256: return bgreen(s)
    if tile >=  128: return green(s)
    if tile >=   64: return cyan(s)
    return white(s)

_session_best: int = 0
def _score_color(s: float, w: int = 9) -> str:
    global _session_best
    fmt = f"{int(s):>{w},}"
    if s >= _session_best: _session_best = int(s); return byellow(fmt)
    if s > _session_best * 0.75: return yellow(fmt)
    return white(fmt)


# ─────────────────────────────── table row ────────────────────────────────────
def _fmt_row(ep_lo: int, ep_hi: int, scores: list[int], moves: list[int],
             tiles: list[int], eps: float, buf: int, loss: float) -> str:
    ep  = cyan(f"{ep_lo:>6,}") + dim(" – ") + cyan(f"{ep_hi:<6,}")
    asc = _score_color(float(np.mean(scores)), W_ASC)
    msc = bwhite(f"{max(scores):>{W_MSC},}")
    mv  = bblue(f"{float(np.mean(moves)):>{W_MV}.1f}")
    med = _tile_color(int(np.median(tiles)), 5)
    mx  = _tile_color(max(tiles), 5)
    e   = magenta(f"{eps:>{W_EPS}.4f}")
    b   = dim(f"{buf:>{W_BUF},}")
    lo  = dim(f"{loss:>{W_LOSS}.4f}") if loss else dim(f"{'—':>{W_LOSS}}")
    return ep + _SEP + asc + _SEP + msc + _SEP + mv + _SEP + med + " " + mx + _SEP + e + _SEP + b + _SEP + lo


# ─────────────────────────────── startup banner ───────────────────────────────
def _startup_banner(device: str, n_params: int, n_episodes: int, lr: float,
                    learn_every: int, eps_decay: float, eps_end: float,
                    models_dir: str, cfg: RewardConfig) -> str:
    def kv(key: str, val: str, note: str = "") -> str:
        content = f"  {bwhite(key.ljust(20))}{bgreen(val)}"
        if note: content += dim(f"  {note}")
        return _box_row(content)

    lines = [
        "",
        bwhite("  ╔" + "═" * BOX_IN + "╗"),
        _box_row(bcyan("  2048  ·  Deep Q-Network  Trainer".center(BOX_IN - 2))),
        _box_blank(),
        kv("device",      device),
        kv("parameters",  f"{n_params:,}"),
        kv("episodes",    f"{n_episodes:,}"),
        _box_blank(),
        _box_row(dim("  Hyperparameters".center(BOX_IN - 2))),
        _box_blank(),
        kv("lr",          str(lr),          "← 3e-4 recommended"),
        kv("learn_every", str(learn_every), "← 4 recommended  (was 25)"),
        kv("eps_decay",   str(eps_decay),   "← 0.9995 recommended  (was 0.9998)"),
        kv("eps_end",     str(eps_end),     "← 0.02 recommended  (was 0.05)"),
        kv("models dir",  models_dir),
        _box_blank(),
        _box_row(dim("  Reward config".center(BOX_IN - 2))),
        _box_blank(),
        kv("merge_weight",     str(cfg.merge_weight)),
        kv("log_scale",        str(cfg.log_scale)),
        kv("survival_bonus",   str(cfg.survival_bonus),   "← 0.0 = must earn every point"),
        kv("empty_weight",     str(cfg.empty_weight),     "← rewards board openness"),
        kv("monotone_weight",  str(cfg.monotone_weight),  "← rewards snake ordering"),
        kv("no_merge_penalty", str(cfg.no_merge_penalty), "← punishes passive moves"),
        kv("milestone_weight", str(cfg.milestone_weight), "← bonus for new max tile"),
        _box_blank(),
        bwhite("  ╚" + "═" * BOX_IN + "╝"),
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────── eval banner ──────────────────────────────────
def _eval_banner(ep: int, stats: dict[str, object], elapsed_s: float, cfg: RewardConfig) -> str:
    def stat_line(*pairs: tuple[str, str]) -> str:
        return _box_row("  " + "   ".join(f"{bgreen(k)}: {v}" for k, v in pairs))

    tc: dict[int, int] = stats["tile_counts"]  # type: ignore[assignment]
    tile_bar = "  ".join(
        f"{_tile_color(t, 4)}{dim('×')}{dim(str(n))}"
        for t, n in sorted(tc.items(), reverse=True)[:6]
    )
    cfg_str = (f"merge×{cfg.merge_weight}  survival={cfg.survival_bonus}  "
               f"empty×{cfg.empty_weight}  mono×{cfg.monotone_weight}  "
               f"no_merge_pen={cfg.no_merge_penalty}  milestone×{cfg.milestone_weight}")

    mean_score: float = stats["mean_score"]  # type: ignore[assignment]
    max_score:  int   = stats["max_score"]   # type: ignore[assignment]
    n_games:    int   = stats["n"]           # type: ignore[assignment]
    mean_tile:  float = stats["mean_tile"]   # type: ignore[assignment]
    max_tile:   int   = stats["max_tile"]    # type: ignore[assignment]

    return "\n".join([
        "",
        bwhite("  ╔" + "═" * BOX_IN + "╗"),
        _box_row(bcyan(f"  EVALUATION  @  episode {ep:,}".center(BOX_IN - 2))),
        _box_blank(),
        stat_line(("avg score", byellow(f"{mean_score:>10,.0f}")),
                  ("max score", byellow(f"{max_score:>10,}")),
                  ("games",     white(str(n_games))),
                  ("time",      dim(f"{elapsed_s:.1f}s"))),
        stat_line(("avg tile",  byellow(f"{mean_tile:>8,.1f}")),
                  ("max tile",  _tile_color(max_tile, 5))),
        _box_blank(),
        _box_row(f"  {bgreen('tile dist')}:  {tile_bar}"),
        _box_blank(),
        _box_row(dim(f"  {cfg_str}")),
        bwhite("  ╚" + "═" * BOX_IN + "╝"),
        "",
    ])


def _save_banner(path: str, ep: int) -> str:
    return "  " + bgreen("✔") + dim("  checkpoint → ") + cyan(path) + dim(f"  (ep {ep:,})")


# ─────────────────────────────── training loop ────────────────────────────────
def train(
    n_episodes:       int              = 50_000,
    window:           int              = 25,
    eval_every:       int              = 500,
    eval_n:           int              = 50,
    save_every:       int              = 1_000,
    resume:           str | None       = None,
    models_dir:       str              = "models",
    # hyperparameters
    lr:               float            = 3e-4,
    learn_every:      int              = 4,
    eps_decay:        float            = 0.9995,
    eps_end:          float            = 0.02,
    gamma:            float            = 0.99,
    batch_size:       int              = 512,
    target_sync:      int              = 500,
    warmup:           int              = 2_000,
    # reward
    reward_cfg:       RewardConfig | None = None,
) -> DQNAgent:
    Path(models_dir).mkdir(exist_ok=True)
    cfg = reward_cfg or DEFAULT_REWARD_CFG

    net_params = sum(p.numel() for p in TwoZeroFourEightNet().parameters())
    print(_startup_banner(str(DEVICE), net_params, n_episodes, lr,
                          learn_every, eps_decay, eps_end, models_dir + "/", cfg))

    agent = DQNAgent(
        lr=lr, gamma=gamma, eps_end=eps_end, eps_decay=eps_decay,
        batch_size=batch_size, target_sync=target_sync, warmup=warmup,
        learn_every=learn_every, reward_cfg=cfg,
    )
    start_ep = 1

    if resume:
        agent.load(resume)
        start_ep = agent.episode_count + 1
        print(f"  {bgreen('resumed')} from {cyan(resume)}")
        print(f"  starting at episode {cyan(str(start_ep))}   ε = {magenta(f'{agent.eps:.4f}')}")
        print()

    w_scores: list[int]   = []
    w_moves:  list[int]   = []
    w_tiles:  list[int]   = []
    w_losses: list[float] = []
    w_start_ep = start_ep

    best_eval_score: float = 0.0
    row_count              = 0
    t_train_start          = time.perf_counter()
    t_window_start         = time.perf_counter()

    interrupted = False
    def _sigint(sig: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        print(f"\n  {yellow('⚠')}  interrupt — saving and exiting…")
    signal.signal(signal.SIGINT, _sigint)

    print(HEADER)
    print(DIVIDER)

    for ep in range(start_ep, start_ep + n_episodes):
        if interrupted:
            break

        score, max_tile, steps = agent.run_episode(train=True)
        w_scores.append(score); w_moves.append(steps); w_tiles.append(max_tile)
        if agent.last_loss:
            w_losses.append(agent.last_loss)

        if len(w_scores) >= window:
            if row_count > 0 and row_count % 20 == 0:
                print(); print(HEADER); print(DIVIDER)
            print(_fmt_row(w_start_ep, ep, w_scores, w_moves, w_tiles,
                           agent.eps, len(agent.buffer),
                           float(np.mean(w_losses)) if w_losses else 0.0))
            row_count += 1; w_start_ep = ep + 1
            w_scores.clear(); w_moves.clear(); w_tiles.clear(); w_losses.clear()
            t_window_start = time.perf_counter()

        if ep % eval_every == 0:
            t0 = time.perf_counter()
            stats = agent.evaluate(eval_n)
            elapsed = time.perf_counter() - t0
            print(_eval_banner(ep, stats, elapsed, cfg))
            print(HEADER); print(DIVIDER)
            row_count = 0
            if stats["mean_score"] > best_eval_score:# type: ignore
                best_eval_score = float(stats["mean_score"])# type: ignore
                best_path = os.path.join(models_dir, "best.pt")
                agent.save(best_path)
                print(f"  {bred('★')}  new best {byellow(f'{best_eval_score:,.0f}')} → {cyan(best_path)}")

        if ep % save_every == 0:
            ckpt = os.path.join(models_dir, f"ckpt_ep{ep:06d}.pt")
            agent.save(ckpt)
            print(_save_banner(ckpt, ep))

    final_path = os.path.join(models_dir, "final.pt")
    agent.save(final_path)
    total = time.perf_counter() - t_train_start
    h, m, s = int(total // 3600), int((total % 3600) // 60), int(total % 60)
    print(f"\n{DIVIDER}")
    print(f"  {bgreen('done')}  │  time: {cyan(f'{h:02d}:{m:02d}:{s:02d}')}  │  model: {cyan(final_path)}\n")
    return agent


# ─────────────────────────────── CLI ──────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the 2048 DQN agent  —  all hyperparameters controllable from CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Recommended fresh-start command:
  python train.py --lr 3e-4 --learn-every 4 --eps-decay 0.9995 --eps-end 0.02 \\
                  --survival-bonus 0.0 --empty-weight 0.3 --monotone-weight 0.2 \\
                  --no-merge-penalty 0.5 --milestone-weight 2.0
        """
    )

    # ── run control ────────────────────────────────────────────────────────
    g = p.add_argument_group("run control")
    g.add_argument("-e", "--episodes",   type=int,   default=50_000, metavar="N",
                   help="total training episodes  (default: 50000)")
    g.add_argument("-r", "--resume",     type=str,   default=None,   metavar="PATH",
                   help="resume from checkpoint .pt file")
    g.add_argument("-w", "--window",     type=int,   default=25,     metavar="N",
                   help="print stats every N episodes  (default: 25)")
    g.add_argument("--eval-every",       type=int,   default=500,    metavar="N",
                   help="run evaluation every N episodes  (default: 500)")
    g.add_argument("--eval-n",           type=int,   default=50,     metavar="N",
                   help="games per evaluation  (default: 50)")
    g.add_argument("--save-every",       type=int,   default=1_000,  metavar="N",
                   help="save checkpoint every N episodes  (default: 1000)")
    g.add_argument("--models-dir",       type=str,   default="models", metavar="DIR",
                   help="checkpoint directory  (default: models/)")

    # ── hyperparameters ────────────────────────────────────────────────────
    g = p.add_argument_group("hyperparameters")
    g.add_argument("--lr",               type=float, default=3e-4,   metavar="F",
                   help="learning rate  (default: 3e-4  ← was 2e-3, too high)")
    g.add_argument("--gamma",            type=float, default=0.99,   metavar="F",
                   help="discount factor  (default: 0.99)")
    g.add_argument("--learn-every",      type=int,   default=4,      metavar="N",
                   help="gradient update every N env steps  (default: 4  ← was 25)")
    g.add_argument("--batch-size",       type=int,   default=512,    metavar="N",
                   help="replay batch size  (default: 512)")
    g.add_argument("--buffer-cap",       type=int,   default=100_000,metavar="N",
                   help="replay buffer capacity  (default: 100000)")
    g.add_argument("--target-sync",      type=int,   default=500,    metavar="N",
                   help="copy policy→target every N gradient steps  (default: 500)")
    g.add_argument("--warmup",           type=int,   default=2_000,  metavar="N",
                   help="don't learn until buffer has N transitions  (default: 2000)")

    # ── epsilon schedule ───────────────────────────────────────────────────
    g = p.add_argument_group("epsilon / exploration")
    g.add_argument("--eps-start",        type=float, default=1.0,    metavar="F",
                   help="starting epsilon  (default: 1.0)")
    g.add_argument("--eps-end",          type=float, default=0.02,   metavar="F",
                   help="minimum epsilon  (default: 0.02  ← was 0.05)")
    g.add_argument("--eps-decay",        type=float, default=0.9995, metavar="F",
                   help="epsilon multiplied by this each gradient step\n"
                        "0.9998=slow decay, 0.9995=medium, 0.999=fast  (default: 0.9995)")

    # ── reward shaping ─────────────────────────────────────────────────────
    g = p.add_argument_group("reward shaping")
    g.add_argument("--merge-weight",     type=float, default=1.0,    metavar="F",
                   help="multiplier on merge score  (default: 1.0)")
    g.add_argument("--no-log-scale",     action="store_true",
                   help="disable log1p reward compression  (not recommended)")
    g.add_argument("--survival-bonus",   type=float, default=0.0,    metavar="F",
                   help="flat reward per step alive  (default: 0.0  ← was 1.0)")
    g.add_argument("--empty-weight",     type=float, default=0.3,    metavar="F",
                   help="bonus per free cell after move  (default: 0.3  ← was 0.1)")
    g.add_argument("--monotone-weight",  type=float, default=0.2,    metavar="F",
                   help="bonus for sorted board layout  (default: 0.2  ← was 0.0)")
    g.add_argument("--no-merge-penalty", type=float, default=0.5,    metavar="F",
                   help="penalty when move slides but doesn't merge  (default: 0.5)")
    g.add_argument("--milestone-weight", type=float, default=2.0,    metavar="F",
                   help="bonus × log2(tile) when agent sets new game max  (default: 2.0)")
    return p


# ─────────────────────────────── interactive setup wizard ─────────────────────
def _ask(prompt: str, default, cast=str, valid=None):
    """
    Print a prompt, show the default in brackets, read input.
    Press Enter to accept the default. Validates against `valid` list if given.
    """
    dflt_str = str(default)
    while True:
        try:
            raw = input(f"  {prompt} [{dflt_str}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)
        if raw == "":
            return default
        try:
            val = cast(raw)
        except (ValueError, TypeError):
            print(f"    ✗  expected {cast.__name__}, got '{raw}' — try again")
            continue
        if valid is not None and val not in valid:
            print(f"    ✗  must be one of {valid} — try again")
            continue
        return val


def _ask_bool(prompt: str, default: bool) -> bool:
    dflt_str = "y" if default else "n"
    while True:
        try:
            raw = input(f"  {prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    ✗  enter y or n")


def _wizard() -> dict:
    """
    Walk the user through every training setting interactively.
    Returns a kwargs dict ready to pass to train().
    """
    W = 62
    print()
    print("  ╔" + "═" * W + "╗")
    print("  ║" + "  2048 DQN — Training Setup".center(W) + "║")
    print("  ║" + "  Press Enter to accept the [default]".center(W) + "║")
    print("  ╚" + "═" * W + "╝")
    print()

    # ── resume? ──────────────────────────────────────────────────────────────
    from pathlib import Path as _P
    resume = None
    models_dir = "models"
    ckpts = sorted(_P(models_dir).glob("*.pt")) if _P(models_dir).exists() else []

    print("  ── Checkpoint ──────────────────────────────────────────────")
    if ckpts:
        print(f"  Found {len(ckpts)} checkpoint(s) in models/:")
        for i, c in enumerate(ckpts[-5:], 1):          # show last 5
            print(f"    {i}.  {c.name}")
        if _ask_bool("Resume from the latest checkpoint?", default=False):
            resume = str(ckpts[-1])
            print(f"  ✓  will resume from {ckpts[-1].name}")
        print()
    else:
        print("  No checkpoints found — starting fresh.\n")

    # ── episodes ─────────────────────────────────────────────────────────────
    print("  ── Run Length ──────────────────────────────────────────────")
    n_episodes = _ask("Total episodes to train", default=50_000, cast=int)# type: ignore
    print()

    # ── preset or custom? ─────────────────────────────────────────────────
    print("  ── Settings Mode ───────────────────────────────────────────")
    print("  [1]  Recommended  (best defaults for a fresh run)")
    print("  [2]  Quick test   (5k episodes, fast settings)")
    print("  [3]  Custom       (you set every value)")
    mode = _ask("Choose mode", default=1, cast=int, valid=[1, 2, 3])# type: ignore
    print()

    if mode == 1:
        # ── recommended defaults ─────────────────────────────────────────
        return dict(
            n_episodes       = n_episodes,
            resume           = resume,
            lr               = 3e-4,
            learn_every      = 4,
            eps_decay        = 0.9995,
            eps_end          = 0.02,
            reward_cfg       = RewardConfig(
                merge_weight     = 1.0,
                log_scale        = True,
                survival_bonus   = 0.0,
                empty_weight     = 0.3,
                monotone_weight  = 0.2,
                no_merge_penalty = 0.5,
                milestone_weight = 2.0,
            ),
        )

    if mode == 2:
        # ── quick-test preset ────────────────────────────────────────────
        return dict(
            n_episodes       = min(n_episodes, 5_000),
            eval_every       = 500,
            resume           = resume,
            lr               = 3e-4,
            learn_every      = 4,
            eps_decay        = 0.999,
            eps_end          = 0.05,
            reward_cfg       = RewardConfig(),
        )

    # ── mode 3: fully custom ─────────────────────────────────────────────────
    print("  ── Hyperparameters ─────────────────────────────────────────")
    print("  (learning rate controls how big each gradient step is)")
    lr          = _ask("Learning rate              (recommended: 3e-4)", default=3e-4, cast=float)# type: ignore
    learn_every = _ask("Gradient update every N steps (recommended: 4)", default=4,    cast=int)# type: ignore
    print()

    print("  ── Exploration (Epsilon) ───────────────────────────────────")
    print("  (epsilon = chance of random move; starts high, decays over time)")
    eps_end   = _ask("Min epsilon when fully trained (recommended: 0.02)", default=0.02,  cast=float)# type: ignore
    eps_decay = _ask("Decay rate per step — 0.9998=slow 0.9995=med 0.999=fast", default=0.9995, cast=float)# type: ignore
    print()

    print("  ── Reward Shaping ──────────────────────────────────────────")
    print("  (these shape what the AI cares about during training)")
    print()
    print("  survival_bonus   — flat reward just for surviving each step")
    print("                     0.0 = agent must earn everything from merges")
    survival_bonus   = _ask("Survival bonus per step    (recommended: 0.0)", default=0.0,  cast=float)# type: ignore

    print()
    print("  empty_weight     — bonus per free cell on the board")
    print("                     rewards keeping the board open for future moves")
    empty_weight     = _ask("Empty cell bonus weight    (recommended: 0.3)", default=0.3,  cast=float)# type: ignore

    print()
    print("  monotone_weight  — bonus for having tiles sorted in snake order")
    print("                     rewards the corner strategy the AI naturally learns")
    monotone_weight  = _ask("Monotone ordering bonus    (recommended: 0.2)", default=0.2,  cast=float)# type: ignore

    print()
    print("  no_merge_penalty — penalty for moves that slide but don't merge")
    print("                     punishes passive shuffling moves")
    no_merge_penalty = _ask("No-merge penalty           (recommended: 0.5)", default=0.5,  cast=float)# type: ignore

    print()
    print("  milestone_weight — bonus × log2(tile) when agent hits a new max tile")
    print("                     big reward for achieving 256, 512, 1024 etc for first time")
    milestone_weight = _ask("Milestone tile bonus       (recommended: 2.0)", default=2.0,  cast=float)# type: ignore

    print()
    log_scale = _ask_bool("Use log reward compression? (recommended: yes)", default=True)
    print()

    return dict(
        n_episodes       = n_episodes,
        resume           = resume,
        lr               = lr,
        learn_every      = learn_every,
        eps_decay        = eps_decay,
        eps_end          = eps_end,
        reward_cfg       = RewardConfig(
            merge_weight     = 1.0,
            log_scale        = log_scale,
            survival_bonus   = survival_bonus,
            empty_weight     = empty_weight,
            monotone_weight  = monotone_weight,
            no_merge_penalty = no_merge_penalty,
            milestone_weight = milestone_weight,
        ),
    )


if __name__ == "__main__":
    import sys
    # If any CLI flags are passed, use the argparse path (for power users / scripting)
    if len(sys.argv) > 1:
        args = _build_parser().parse_args()
        train(
            n_episodes  = args.episodes,
            window      = args.window,
            eval_every  = args.eval_every,
            eval_n      = args.eval_n,
            save_every  = args.save_every,
            resume      = args.resume,
            models_dir  = args.models_dir,
            lr          = args.lr,
            gamma       = args.gamma,
            learn_every = args.learn_every,
            batch_size  = args.batch_size,
            target_sync = args.target_sync,
            warmup      = args.warmup,
            eps_decay   = args.eps_decay,
            eps_end     = args.eps_end,
            reward_cfg  = RewardConfig(
                merge_weight     = args.merge_weight,
                log_scale        = not args.no_log_scale,
                survival_bonus   = args.survival_bonus,
                empty_weight     = args.empty_weight,
                monotone_weight  = args.monotone_weight,
                no_merge_penalty = args.no_merge_penalty,
                milestone_weight = args.milestone_weight,
            ),
        )
    else:
        # No flags → interactive wizard
        kwargs = _wizard()
        train(**kwargs)