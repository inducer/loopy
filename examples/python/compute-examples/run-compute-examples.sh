#!/usr/bin/env bash
set -euo pipefail

PYTHON="$(which python)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$SCRIPT_DIR/artifacts"

cd "$SCRIPT_DIR"
mkdir -p "$ARTIFACT_DIR"

run_example() {
    echo
    echo "===== $* ====="
    "$PYTHON" "$@"
}

run_example finite-difference-2-5D.py \
    --npoints 160 \
    --stencil-width 9 \
    --compare \
    --run-kernel \
    --warmup 2 \
    --iterations 5 \
    --json-report "$ARTIFACT_DIR/finite-difference-2-5d.json"

run_example wave-equation-ring-buffer.py \
    --ntime 4194304 \
    --compare \
    --run-kernel \
    --warmup 2 \
    --iterations 8 \
    --json-report "$ARTIFACT_DIR/wave-equation-ring-buffer.json"

run_example matmul.py \
    --m 1024 \
    --n 1024 \
    --k 1024 \
    --bm 64 \
    --bn 64 \
    --bk 16 \
    --tm 4 \
    --tn 4 \
    --compare \
    --warmup 3 \
    --iterations 8 \
    --json-report "$ARTIFACT_DIR/matmul.json"

run_example l2p-tiled-basis-compute.py \
    --ntargets 8192 \
    --order 16 \
    --target-block-size 64 \
    --compare \
    --run-kernel \
    --warmup 2 \
    --iterations 5 \
    --json-report "$ARTIFACT_DIR/l2p-tiled-basis-compute.json"

run_example p2m-basis-compute.py \
    --nsources 8192 \
    --order 16 \
    --q1-tile-size 17 \
    --source-tile-size 256 \
    --compare \
    --run-kernel \
    --warmup 2 \
    --iterations 5 \
    --json-report "$ARTIFACT_DIR/p2m-basis-compute.json"

run_example m2m-sum-factorization.py \
    --order 31 \
    --eta-tile-size 8 \
    --dimension 3 \
    --compare \
    --run-kernel \
    --warmup 2 \
    --iterations 5 \
    --json-report "$ARTIFACT_DIR/m2m-sum-factorization.json"

run_example l2p-3d-tensor-product-compute.py \
    --ntargets 4096 \
    --order 12 \
    --target-block-size 64 \
    --compare \
    --run-kernel \
    --warmup 2 \
    --iterations 5 \
    --json-report "$ARTIFACT_DIR/l2p-3d-tensor-product-compute.json"
