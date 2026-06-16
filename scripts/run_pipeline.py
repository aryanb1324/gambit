#!/usr/bin/env python3
"""
Gambit — Full Autonomous Pipeline
Phases 1 → 2 → 3, runs without human input.
Logs to data/pipeline.log. Sends Discord progress pings via OpenClaw.
"""
from __future__ import annotations
import sys, os, json, time, subprocess, signal, textwrap
sys.path.insert(0, '/Users/dev1/.openclaw/workspace/gambit')

# ── Paths ────────────────────────────────────────────────────────────────────
GAMBIT_DIR  = '/Users/dev1/.openclaw/workspace/gambit'
DATA_DIR    = os.path.join(GAMBIT_DIR, 'data')
CKPT_DIR    = os.path.join(GAMBIT_DIR, 'checkpoints')
ZST_FILE    = os.path.join(DATA_DIR,  'lichess_2024-01.pgn.zst')
PGN_FILE    = os.path.join(DATA_DIR,  'lichess_2024-01.pgn')
H5_FILE     = os.path.join(DATA_DIR,  'lichess_2024-01_2200elo.h5')
LOG_FILE    = os.path.join(DATA_DIR,  'pipeline.log')
STOCKFISH   = '/opt/homebrew/bin/stockfish'
ARYAN_ID    = '1511142674780917852'

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('gambit')

# ── Discord ping via OpenClaw CLI ─────────────────────────────────────────────
def ping(msg: str):
    """Send a Discord DM to Aryan via openclaw CLI."""
    try:
        subprocess.run(
            ['openclaw', 'message', '--to', ARYAN_ID, '--text', msg],
            capture_output=True, timeout=10
        )
    except Exception as e:
        log.warning(f'Discord ping failed: {e}')

# ── Phase 1a: Decompress ──────────────────────────────────────────────────────
def phase1a_decompress():
    if os.path.exists(PGN_FILE):
        log.info('Phase 1a: PGN already exists, skipping decompression.')
        return
    log.info(f'Phase 1a: Decompressing {ZST_FILE} ...')
    ping('🎮 **Gambit Phase 1a** — Decompressing Lichess database (32GB). This will take ~15 min.')
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    t0 = time.time()
    with open(ZST_FILE, 'rb') as ifh, open(PGN_FILE, 'wb') as ofh:
        dctx.copy_stream(ifh, ofh, write_size=1 << 22)
    elapsed = (time.time() - t0) / 60
    size_gb = os.path.getsize(PGN_FILE) / 1e9
    log.info(f'Phase 1a: Done. {size_gb:.1f} GB PGN in {elapsed:.1f} min.')

# ── Phase 1b: Parse ───────────────────────────────────────────────────────────
def phase1b_parse():
    if os.path.exists(H5_FILE):
        log.info('Phase 1b: HDF5 already exists, skipping parse.')
        return
    log.info('Phase 1b: Parsing PGN → HDF5 (2200+ Elo, 15+ moves) ...')
    ping('🎮 **Gambit Phase 1b** — Parsing games (filtering to 2200+ Elo). ETA: 20-40 min.')
    from gambit.data.parser import parse_pgn_to_tensors
    t0 = time.time()
    n = parse_pgn_to_tensors(PGN_FILE, H5_FILE, min_elo=2200, min_moves=15)
    elapsed = (time.time() - t0) / 60
    size_mb = os.path.getsize(H5_FILE) / 1e6
    log.info(f'Phase 1b: Done. {n:,} positions, {size_mb:.0f} MB HDF5 in {elapsed:.1f} min.')
    ping(f'🎮 **Gambit Phase 1b** — Done! {n:,} training positions saved. ({size_mb:.0f} MB)')

# ── Phase 1c: Supervised Training ─────────────────────────────────────────────
def phase1c_train() -> str:
    """Returns path to best checkpoint."""
    import torch
    from gambit.train.supervised import train_supervised

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    log.info(f'Phase 1c: Supervised training on {device} ...')
    ping(f'🎮 **Gambit Phase 1c** — Supervised training starting on {device} (Apple M4 MPS). ETA: 1-3 hrs.')

    t0 = time.time()
    train_supervised(
        data_path      = H5_FILE,
        checkpoint_dir = CKPT_DIR,
        num_epochs     = 5,
        batch_size     = 256,
        lr             = 1e-3,
        weight_decay   = 1e-4,
        num_res_blocks = 10,
        num_filters    = 128,
        device         = device,
    )
    elapsed = (time.time() - t0) / 60

    # Find best checkpoint
    ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.endswith('.pt')])
    best  = os.path.join(CKPT_DIR, ckpts[-1]) if ckpts else None
    log.info(f'Phase 1c: Done in {elapsed:.1f} min. Best checkpoint: {best}')
    ping(f'🎮 **Gambit Phase 1c** — Supervised training complete! ({elapsed:.0f} min)\nCheckpoint: {best}\n\nStarting Phase 2 — MCTS arena test.')
    return best

# ── Phase 2: MCTS + Arena vs Stockfish ────────────────────────────────────────
def phase2_arena(checkpoint_path: str) -> dict:
    import torch
    from gambit.network.resnet import GambitNet, load_checkpoint
    from gambit.mcts.mcts import MCTS
    from gambit.arena.vs_stockfish import run_arena

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    log.info(f'Phase 2: Loading checkpoint {checkpoint_path} ...')
    model, _, _ = load_checkpoint(checkpoint_path, device)
    model.eval()

    results = {}
    for elo in [1500, 2000, 2500]:
        log.info(f'Phase 2: Arena vs Stockfish Elo {elo} (20 games) ...')
        mcts = MCTS(model, c_puct=1.0, num_simulations=200, device=device)
        r = run_arena(
            model          = model,
            mcts           = mcts,
            num_games      = 20,
            stockfish_path = STOCKFISH,
            stockfish_elo  = elo,
        )
        results[elo] = r
        log.info(f'  Elo {elo}: W{r["wins"]} D{r["draws"]} L{r["losses"]} '
                 f'({r["win_rate"]*100:.0f}% WR, est. Elo ~{r["elo_estimate"]:.0f})')

    # Build report
    lines = ['🎮 **Gambit Phase 2 — Arena Results (post-supervised, pre-RL)**']
    for elo, r in results.items():
        lines.append(f'vs Stockfish {elo}: W{r["wins"]} D{r["draws"]} L{r["losses"]} '
                     f'| WR {r["win_rate"]*100:.0f}% | Est. Elo ~{r["elo_estimate"]:.0f}')
    lines.append('Starting Phase 3 — Self-play RL loop.')
    ping(' | '.join(lines))

    # Save results
    results_path = os.path.join(CKPT_DIR, 'phase2_arena_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f'Phase 2: Results saved to {results_path}')
    return results

# ── Phase 3: Self-Play RL ─────────────────────────────────────────────────────
def phase3_selfplay(checkpoint_path: str):
    import torch
    from gambit.train.selfplay import train_selfplay

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    log.info(f'Phase 3: Self-play RL starting from {checkpoint_path} on {device} ...')
    ping('🎮 **Gambit Phase 3** — Self-play RL loop starting. Will run 50 iterations '
         '(~25 games + training per iteration). Progress pings every 10 iterations.')

    train_selfplay(
        checkpoint_path     = checkpoint_path,
        output_dir          = CKPT_DIR,
        num_iterations      = 50,
        games_per_iter      = 25,
        train_steps_per_iter= 200,
        batch_size          = 256,
        num_simulations     = 200,
        device              = device,
        progress_callback   = _selfplay_progress_cb,
    )
    log.info('Phase 3: Self-play RL complete.')
    ping('🎮 **Gambit Phase 3 complete!** Self-play RL finished 50 iterations. '
         'Final checkpoints in `gambit/checkpoints/`. Run Phase 2 arena again to measure improvement.')

def _selfplay_progress_cb(iteration: int, total: int, stats: dict):
    """Called by train_selfplay every 10 iterations."""
    if iteration % 10 == 0:
        msg = (f'🎮 **Gambit Phase 3** — Iteration {iteration}/{total} | '
               f'P-loss {stats.get("policy_loss", "?"):.3f} | '
               f'V-loss {stats.get("value_loss", "?"):.3f} | '
               f'Avg game len {stats.get("avg_game_length", "?"):.0f} moves')
        log.info(msg)
        ping(msg)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log.info('=' * 60)
    log.info('GAMBIT — Full Autonomous Pipeline Starting')
    log.info('=' * 60)
    ping('🎮 **Gambit pipeline started!** Running Phases 1 → 2 → 3 autonomously. Will ping with progress.')

    try:
        # Phase 1
        phase1a_decompress()
        phase1b_parse()
        checkpoint = phase1c_train()

        if not checkpoint:
            log.error('No checkpoint found after Phase 1c. Aborting.')
            ping('❌ **Gambit** — No checkpoint after Phase 1c training. Check `data/pipeline.log`.')
            sys.exit(1)

        # Phase 2
        phase2_arena(checkpoint)

        # Phase 3
        phase3_selfplay(checkpoint)

        log.info('=' * 60)
        log.info('GAMBIT — All phases complete.')
        log.info('=' * 60)
        ping('✅ **Gambit — All phases done!** Pipeline complete. Check `gambit/checkpoints/` for the final model.')

    except KeyboardInterrupt:
        log.info('Pipeline interrupted by user.')
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log.error(f'Pipeline failed: {e} | {err}')
        ping(f'❌ **Gambit pipeline error:** {e} — Check data/pipeline.log for full traceback.')
        sys.exit(1)
