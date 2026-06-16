#!/usr/bin/env bash
# download_lichess.sh — download and decompress a Lichess monthly PGN archive
#
# Usage:
#   ./scripts/download_lichess.sh 2024-01
#   ./scripts/download_lichess.sh 2024-01 --no-decompress
#
# The script downloads:
#   https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst
# and decompresses it to data/lichess_db_standard_rated_YYYY-MM.pgn

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <YYYY-MM> [--no-decompress]"
    exit 1
fi

YEAR_MONTH="$1"
DECOMPRESS=true

for arg in "$@"; do
    [[ "$arg" == "--no-decompress" ]] && DECOMPRESS=false
done

FILENAME="lichess_db_standard_rated_${YEAR_MONTH}.pgn.zst"
BASE_URL="https://database.lichess.org/standard"
URL="${BASE_URL}/${FILENAME}"
OUT_DIR="data"
ZST_PATH="${OUT_DIR}/${FILENAME}"
PGN_PATH="${ZST_PATH%.zst}"

mkdir -p "$OUT_DIR"

if [[ -f "$ZST_PATH" ]]; then
    echo "[download] Already exists: $ZST_PATH"
else
    echo "[download] Fetching $URL ..."
    curl -L --progress-bar -o "$ZST_PATH" "$URL"
    echo "[download] Saved to $ZST_PATH"
fi

if $DECOMPRESS; then
    if [[ -f "$PGN_PATH" ]]; then
        echo "[decompress] Already exists: $PGN_PATH"
    else
        echo "[decompress] Decompressing $ZST_PATH ..."
        if command -v zstd &>/dev/null; then
            zstd -d "$ZST_PATH" -o "$PGN_PATH"
        else
            echo "[decompress] zstd not found — falling back to Python zstandard"
            python3 -c "
from gambit.data.download import decompress_zst
decompress_zst('$ZST_PATH', '$PGN_PATH')
"
        fi
        echo "[decompress] Done: $PGN_PATH"
    fi
fi

echo "Finished. Files:"
ls -lh "${OUT_DIR}/"lichess_db_standard_rated_${YEAR_MONTH}* 2>/dev/null || true
