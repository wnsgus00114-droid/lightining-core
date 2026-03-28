#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT_DIR/build/benchmarks/bench_matrix_ops"
OUT_DIR="$ROOT_DIR/build/benchmarks"
OUT_CSV="$OUT_DIR/matrix_ops_sweep.csv"

if [[ ! -x "$BIN" ]]; then
  echo "bench_matrix_ops binary not found. Build first: cmake --build build -j"
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "rows,cols,batch,cpu_sub,metal_sub_off,metal_sub_on,speedup_sub_on,cpu_div,metal_div_off,metal_div_on,speedup_div_on" > "$OUT_CSV"

rows_list=(512 1024 2048 4096 8192)
cols_list=(128 256 512 1024)
batch_list=(1 2 4)

for rows in "${rows_list[@]}"; do
  for cols in "${cols_list[@]}"; do
    for batch in "${batch_list[@]}"; do
      out=$(CJ_ME_ROWS="$rows" CJ_ME_COLS="$cols" CJ_ME_BATCH="$batch" CJ_ME_WARMUP=6 CJ_ME_ITERS=24 "$BIN")

      cpu_sub=$(echo "$out" | awk -F': ' '/CPU matrixSub/{print $2}' | awk '{print $1}')
      metal_sub_off=$(echo "$out" | awk -F': ' '/Metal matrixSub \(resident=off\)/{print $2}' | awk '{print $1}')
      metal_sub_on=$(echo "$out" | awk -F': ' '/Metal matrixSub \(resident=on\)/{print $2}' | awk '{print $1}')
      speedup_sub_on=$(echo "$out" | awk -F': ' '/Speedup matrixSub on \(CPU\/Metal\)/{print $2}' | sed 's/x$//')

      cpu_div=$(echo "$out" | awk -F': ' '/CPU matrixDiv/{print $2}' | awk '{print $1}')
      metal_div_off=$(echo "$out" | awk -F': ' '/Metal matrixDiv \(resident=off\)/{print $2}' | awk '{print $1}')
      metal_div_on=$(echo "$out" | awk -F': ' '/Metal matrixDiv \(resident=on\)/{print $2}' | awk '{print $1}')
      speedup_div_on=$(echo "$out" | awk -F': ' '/Speedup matrixDiv on \(CPU\/Metal\)/{print $2}' | sed 's/x$//')

      echo "$rows,$cols,$batch,$cpu_sub,$metal_sub_off,$metal_sub_on,$speedup_sub_on,$cpu_div,$metal_div_off,$metal_div_on,$speedup_div_on" >> "$OUT_CSV"
    done
  done
done

echo "[sweep] csv: $OUT_CSV"

python3 - <<'PY'
import csv
from pathlib import Path

csv_path = Path("build/benchmarks/matrix_ops_sweep.csv")
rows = list(csv.DictReader(csv_path.open()))

sub_wins = [r for r in rows if float(r["speedup_sub_on"]) > 1.0]
div_wins = [r for r in rows if float(r["speedup_div_on"]) > 1.0]

print("[sweep] matrixSub resident-on wins:", len(sub_wins), "/", len(rows))
print("[sweep] matrixDiv resident-on wins:", len(div_wins), "/", len(rows))

if sub_wins:
    best_sub = max(sub_wins, key=lambda r: float(r["speedup_sub_on"]))
    print("[best] matrixSub speedup_on=", best_sub["speedup_sub_on"], "at rows=", best_sub["rows"], "cols=", best_sub["cols"], "batch=", best_sub["batch"])

if div_wins:
    best_div = max(div_wins, key=lambda r: float(r["speedup_div_on"]))
    print("[best] matrixDiv speedup_on=", best_div["speedup_div_on"], "at rows=", best_div["rows"], "cols=", best_div["cols"], "batch=", best_div["batch"])
PY
