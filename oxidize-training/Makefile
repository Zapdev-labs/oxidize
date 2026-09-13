# oxidize-training: from-scratch GPT trainer (CPU OpenBLAS)
OPENBLAS ?= $(HOME)/opt/openblas
CC ?= cc
CFLAGS ?= -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS_OB = -I$(OPENBLAS)/include $(OPENBLAS)/lib/libopenblas.a -lm -lpthread

.PHONY: all bench corpus clean train-bin eval backtest news market-corpus

all: bin/gen_corpus bin/gemm_bench bin/bpe bin/oxidize-training

bin:
	mkdir -p bin

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

corpus: bin/gen_corpus
	./bin/gen_corpus 80000 data/corpus.txt

eval: bin/oxidize-training
	chmod +x tools/eval_actions.sh
	./tools/eval_actions.sh ./bin/oxidize-training out/trader.bin data/vocab.bin

backtest: bin/oxidize-training
	chmod +x tools/fetch_ohlc.sh
	./bin/oxidize-training backtest --ckpt out/trader.bin --vocab data/vocab.bin --csv data/spy.csv --symbol SPY --out out/bt-spy.tsv

news:
	python3 tools/fetch_news.py --csv data/spy.csv --out-dir data/news

picks: bin/oxidize-training
	./bin/oxidize-training picks --ckpt out/trader.bin --vocab data/vocab.bin --data-dir data --news-dir data/news

clean:
	rm -rf bin
