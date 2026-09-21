# oxidize-training: from-scratch GPT trainer (CPU OpenBLAS)
OPENBLAS ?= $(HOME)/opt/openblas
CC ?= cc
CFLAGS ?= -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS_OB = -I$(OPENBLAS)/include $(OPENBLAS)/lib/libopenblas.a -lm -lpthread

.PHONY: all bench corpus clean train-bin train eval backtest news market-corpus picks tokens

all: bin/gen_corpus bin/gemm_bench bin/bpe bin/oxidize-training

train-bin: bin/oxidize-training

bin:
	mkdir -p bin

data:
	mkdir -p data

bin/gen_corpus: tools/gen_corpus.c | bin
	$(CC) $(CFLAGS) -o $@ $<

bin/gemm_bench: bench/gemm_bench.c | bin
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS_OB)

bin/bpe: tools/bpe.c | bin
	$(CC) $(CFLAGS) -o $@ $<

bin/oxidize-training: src/train_gpt.c | bin
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS_OB)

bench: bin/gemm_bench
	OMP_NUM_THREADS=8 ./bin/gemm_bench 256 1024 384 8

corpus: data/corpus.txt

data/corpus.txt: bin/gen_corpus | data
	./bin/gen_corpus 80000 $@

market-corpus: | data
	python3 tools/gen_market_corpus.py --data-dir data --news-dir data/news --out data/corpus.txt

# The trainer reads tokens.bin and every scoring command reads vocab.bin, so the
# BPE step has to sit between the corpus and training.
tokens: data/tokens.bin

data/vocab.bin: bin/bpe data/corpus.txt
	./bin/bpe train data/corpus.txt $@

data/tokens.bin: bin/bpe data/corpus.txt data/vocab.bin
	./bin/bpe encode data/corpus.txt data/vocab.bin $@

train: bin/oxidize-training data/tokens.bin
	./bin/oxidize-training train --tokens data/tokens.bin --out out/trader.bin

eval: bin/oxidize-training data/vocab.bin
	chmod +x tools/eval_actions.sh
	./tools/eval_actions.sh ./bin/oxidize-training out/trader.bin data/vocab.bin

backtest: bin/oxidize-training data/vocab.bin
	chmod +x tools/fetch_ohlc.sh
	./bin/oxidize-training backtest --ckpt out/trader.bin --vocab data/vocab.bin --data-dir data --csv data/spy.csv --symbol SPY --out out/bt-spy.tsv

news: | data
	python3 tools/fetch_news.py --csv data/spy.csv --out-dir data/news

picks: bin/oxidize-training data/vocab.bin
	./bin/oxidize-training picks --ckpt out/trader.bin --vocab data/vocab.bin --data-dir data --news-dir data/news

clean:
	rm -rf bin
