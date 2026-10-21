#!/usr/bin/env bash
set -euo pipefail
DEST="${1:?usage: archive_l63_artifacts.sh <dest_dir>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$DEST"
tar -C "$ROOT" -czf "$DEST/l63_table2_artifacts.tar.gz" \
    experiments/l63_norm_stats.pt \
    experiments/L63B_directunet_monaiM/resolved_config.yaml \
    experiments/L63B_directunet_monaiM/checkpoints/stage1_best.ckpt \
    experiments/L63B_vanillacfm_monaiM/resolved_config.yaml \
    experiments/L63B_vanillacfm_monaiM/checkpoints/stage1_best.ckpt \
    experiments/L63B_sda2_monaiM/resolved_config.yaml \
    experiments/L63B_sda2_monaiM/checkpoints/stage1_best.ckpt \
    experiments/l63_da_table2
(cd "$DEST" && sha256sum l63_table2_artifacts.tar.gz | tee l63_table2_artifacts.sha256)
