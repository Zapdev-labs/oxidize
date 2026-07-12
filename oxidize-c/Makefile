CC ?= cc
CFLAGS ?= -O3 -march=native -flto -fopenmp -Wall -Wextra -std=c11 -D_GNU_SOURCE
LDLIBS = -lm

SRC = quant.c kernels.c gguf.c tokenizer.c model.c gen.c server.c prune.c finetune.c main.c

oxidize-c: $(SRC) oc.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

test: oxidize-c test_oc.c
	$(CC) $(CFLAGS) -o test_oc quant.c kernels.c gguf.c tokenizer.c model.c prune.c finetune.c test_oc.c $(LDLIBS)
	./test_oc

clean:
	rm -f oxidize-c test_oc

.PHONY: test clean

CUDA_HOME ?= /usr/local/cuda
cuda: $(SRC) cuda.cu oc.h
	nvcc -O3 -arch=native -c cuda.cu -o cuda.o 2>/dev/null || nvcc -O3 -c cuda.cu -o cuda.o
	$(CC) $(CFLAGS) -DOC_CUDA -o oxidize-c-cuda $(SRC) cuda.o -L$(CUDA_HOME)/lib64 -lcublas -lcudart -lstdc++ $(LDLIBS)
