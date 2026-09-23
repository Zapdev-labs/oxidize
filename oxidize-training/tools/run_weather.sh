#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 tools/fetch_climate.py --out-dir data/climate
python3 tools/gen_weather_corpus.py --climate-dir data/climate --out-dir data/weather

make -j bin/bpe bin/oxidize-training

./bin/bpe train data/weather/corpus.txt data/weather/vocab.bin 4096
./bin/bpe encode data/weather/corpus.txt data/weather/vocab.bin data/weather/tokens.bin

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

./bin/oxidize-training train \
  --tokens data/weather/tokens.bin \
  --out out/weather.bin \
  --steps 1800 \
  --batch 8 \
  --seq 128 \
  --n-layer 6 \
  --n-head 8 \
  --n-embd 256 \
  --lr 3e-4 \
  2>&1 | tee out/weather-train.log

python3 tools/eval_weather.py \
  --bin ./bin/oxidize-training \
  --ckpt out/weather.bin \
  --vocab data/weather/vocab.bin \
  --test data/weather/test.jsonl \
  --forecast data/weather/forecast_2027.jsonl \
  --out out/weather
