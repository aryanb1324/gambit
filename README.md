# Gambit ♟️

**Gambit** is an AlphaZero-style chess engine built in Python. It learns entirely from games and self-play — no hand-crafted evaluation functions. The goal: beat Stockfish.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        GAMBIT                           │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Lichess │───▶│  Supervised  │───▶│  Self-Play RL │  │
│  │  PGN DB  │    │  Pretraining │    │     Loop      │  │
│  └──────────┘    └──────────────┘    └───────┬───────┘  │
│                                              │           │
│  ┌───────────────────────────────────────────▼─────────┐ │
│  │                  GambitNet (ResNet)                  │ │
│  │   Input: (17, 8, 8) board tensor                    │ │
│  │   ┌─────────────────────────────────────────────┐   │ │
│  │   │  Conv(128) → BN → ReLU                      │   │ │
│  │   │  ResBlock × 10                              │   │ │
│  │   └───────────────┬─────────────────────────────┘   │ │
│  │                   │                                 │ │
│  │          ┌────────┴─────────┐                       │ │
│  │          ▼                  ▼                       │ │
│  │   Policy Head         Value Head                    │ │
│  │   (4096 moves)        (scalar [-1,1])               │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                  MCTS (800 sims)                    │ │
│  │  PUCT selection → Neural eval → Backup              │ │
│  │  + Dirichlet noise at root                          │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Training Pipeline (3 Phases)

### Phase 1 — Supervised Pretraining
Train on Lichess games (ELO ≥ 2200) to give the network a strong opening prior.

```
Lichess PGN → parse_pgn_to_tensors() → HDF5 → train_supervised()
```

### Phase 2 — MCTS Integration
Combine the trained network with Monte Carlo Tree Search. The policy head provides priors; the value head replaces rollouts.

### Phase 3 — Self-Play RL
AlphaZero-style self-play: generate games using MCTS, accumulate positions in a replay buffer, train on MCTS visit-count targets. Iterate.

```
model → MCTS games → replay buffer → train → better model → repeat
```

---

## Quickstart

### Install

```bash
git clone https://github.com/your-org/gambit
cd gambit
pip install -e .
```

### Download Lichess Data

```bash
# Download and decompress Jan 2024
./scripts/download_lichess.sh 2024-01
```

Or from Python:

```python
from gambit.data.download import download_month
download_month("2024-01", "data/")
```

### Parse to Training Tensors

```bash
python -c "
from gambit.data.parser import parse_pgn_to_tensors
n = parse_pgn_to_tensors('data/lichess_db_standard_rated_2024-01.pgn', 'data/lichess_2024-01.h5', min_elo=2200)
print(f'Saved {n} positions')
"
```

### Train (Supervised)

```bash
python scripts/train_supervised.py \
  --data data/lichess_2024-01.h5 \
  --epochs 10 \
  --checkpoint-dir checkpoints/
```

### Train (Self-Play)

```bash
python scripts/train_selfplay.py \
  --checkpoint checkpoints/supervised_final.pt \
  --output checkpoints/ \
  --iterations 100
```

### Play

```bash
# Interactive terminal game
python scripts/play.py --checkpoint checkpoints/selfplay_latest.pt

# vs Stockfish at ELO 2000
python scripts/play.py --checkpoint checkpoints/selfplay_latest.pt --vs-stockfish --elo 2000
```

### Run as UCI Engine

```bash
python -m gambit.engine.uci --checkpoint checkpoints/selfplay_latest.pt
```

---

## Hardware Requirements

- **Minimum:** CPU-only works for small experiments
- **Recommended:** NVIDIA GPU with 8+ GB VRAM
- **Fleet hardware:**
  - **Sparq** — RTX 5090 (primary training node)
  - **Buster** — RTX 3080 Ti (secondary / parallel self-play)

Self-play can be distributed: run `selfplay.py` on multiple nodes, merge replay buffers, train on Sparq.

---

## Project Structure

```
gambit/
├── gambit/
│   ├── board/encoder.py      # Board → (17,8,8) tensor + move encoding
│   ├── network/resnet.py     # GambitNet (ResNet policy+value)
│   ├── mcts/mcts.py          # Monte Carlo Tree Search
│   ├── engine/uci.py         # UCI protocol interface
│   ├── data/
│   │   ├── download.py       # Lichess DB downloader
│   │   └── parser.py         # PGN → HDF5 training tensors
│   ├── train/
│   │   ├── supervised.py     # Phase 1: supervised training
│   │   └── selfplay.py       # Phase 3: self-play RL
│   └── arena/
│       └── vs_stockfish.py   # Gambit vs Stockfish evaluation
├── scripts/                  # CLI entry points
└── tests/                    # pytest suite
```

---

## License

MIT
