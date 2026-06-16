#!/usr/bin/env python3
"""Phase 1 launcher: parse Lichess data then kick off supervised training."""
import sys, os, time
sys.path.insert(0, '/Users/dev1/.openclaw/workspace/gambit')

DATA_DIR     = '/Users/dev1/.openclaw/workspace/gambit/data'
CKPT_DIR     = '/Users/dev1/.openclaw/workspace/gambit/checkpoints'
ZST_FILE     = os.path.join(DATA_DIR, 'lichess_2024-01.pgn.zst')
PGN_FILE     = os.path.join(DATA_DIR, 'lichess_2024-01.pgn')
H5_FILE      = os.path.join(DATA_DIR, 'lichess_2024-01_2200elo.h5')

os.makedirs(CKPT_DIR, exist_ok=True)

# --- Step 1: decompress ---
if not os.path.exists(PGN_FILE):
    print(f'[phase1] Decompressing {ZST_FILE} ...', flush=True)
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with open(ZST_FILE, 'rb') as ifh, open(PGN_FILE, 'wb') as ofh:
        dctx.copy_stream(ifh, ofh, write_size=1 << 22)
    print(f'[phase1] Decompressed → {PGN_FILE}', flush=True)
else:
    print(f'[phase1] PGN already exists, skipping decompression.', flush=True)

# --- Step 2: parse ---
if not os.path.exists(H5_FILE):
    print('[phase1] Parsing PGN → HDF5 (2200+ Elo) ...', flush=True)
    from gambit.data.parser import parse_pgn_to_tensors
    n = parse_pgn_to_tensors(PGN_FILE, H5_FILE, min_elo=2200, min_moves=15)
    print(f'[phase1] Parsed {n:,} positions → {H5_FILE}', flush=True)
else:
    print(f'[phase1] HDF5 already exists, skipping parse.', flush=True)

# --- Step 3: supervised training ---
print('[phase1] Starting supervised training on MPS ...', flush=True)
import torch
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'[phase1] Device: {device}', flush=True)

from gambit.train.supervised import train_supervised
train_supervised(
    data_path     = H5_FILE,
    checkpoint_dir= CKPT_DIR,
    num_epochs    = 5,
    batch_size    = 256,
    lr            = 1e-3,
    num_res_blocks= 10,
    num_filters   = 128,
    device        = device,
)
print('[phase1] Training complete. Checkpoints at:', CKPT_DIR, flush=True)
