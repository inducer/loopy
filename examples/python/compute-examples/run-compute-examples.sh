#!/usr/bin/env bash
set -euo pipefail

PYTHON="$(which python)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

run_example() {
    echo
    echo "===== $* ====="
    "$PYTHON" "$@"
}

run_example finite-difference-2-5D.py \
    --npoints 96 \
    --stencil-width 9 \
    --compute \
    --run-kernel \
    --warmup 2 \
    --iterations 3

run_example wave-equation-ring-buffer.py \
    --ntime 4096 \
    --compute \
    --run-kernel \
    --warmup 2 \
    --iterations 5

run_example matmul.py \
    --m 512 \
    --n 512 \
    --k 512 \
    --bm 32 \
    --bn 32 \
    --bk 16 \
    --shared-memory-tiled

run_example matmul.py \
    --m 512 \
    --n 512 \
    --k 512 \
    --bm 64 \
    --bn 64 \
    --bk 16 \
    --tm 4 \
    --tn 4 \
    --register-tiled

run_example l2p-tiled-basis-compute.py \
    --ntargets 512 \
    --order 12 \
    --target-block-size 64 \
    --compare \
    --run-kernel \
    --warmup 2 \
    --iterations 3

run_example p2m-basis-compute.py \
    --nsources 512 \
    --order 12 \
    --q1-tile-size 13 \
    --source-tile-size 128 \
    --compare \
    --run-kernel \
    --warmup 2 \
    --iterations 3

run_example m2m-sum-factorization.py \
    --order 23 \
    --eta-tile-size 8 \
    --dimension 3 \
    --compare \
    --run-kernel \
    --warmup 2 \
    --iterations 3

run_example l2p-3d-tensor-product-compute.py \
    --ntargets 512 \
    --order 8 \
    --target-block-size 64 \
    --compare \
    --run-kernel \
    --warmup 2 \
    --iterations 3
