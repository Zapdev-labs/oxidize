CC ?= cc
CFLAGS ?= -O3 -march=native -flto -fopenmp -Wall -Wextra -std=c11 -D_GNU_SOURCE
LDLIBS = -lm

SRC = quant.c kernels.c numa.c gguf.c tokenizer.c model.c gen.c server.c prune.c finetune.c batch.c main.c

oxidize-c: $(SRC) oc.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

oc-merge: merge.c
	$(CC) $(CFLAGS) -o $@ merge.c $(LDLIBS)

merge: oc-merge

TEST_OUTPUT = $(if $(TEST_BIN),$(TEST_BIN),test_oc)
TEST_RUN = $(if $(TEST_BIN),$(TEST_BIN),./test_oc)
TEST_SRC = quant.c kernels.c numa.c gguf.c tokenizer.c model.c gen.c prune.c finetune.c batch.c test_oc.c

RAW_TPS_SRC = quant.c kernels.c numa.c gguf.c tokenizer.c model.c raw_tps_bench.c
RAW_TPS_TEST_BIN = $(if $(TEST_BIN),$(TEST_BIN),raw-tps-bench-test)
RAW_TPS_TEST_RUN = $(if $(TEST_BIN),$(TEST_BIN),./raw-tps-bench-test)

test: $(TEST_SRC) oc.h
	@mkdir -p $(dir $(TEST_OUTPUT))
	$(CC) $(CFLAGS) -o $(TEST_OUTPUT) $(TEST_SRC) $(LDLIBS)
	$(TEST_RUN)

raw-tps-bench: $(RAW_TPS_SRC) oc.h
	$(CC) $(CFLAGS) -o $@ $(RAW_TPS_SRC) $(LDLIBS)

raw-tps-bench-test: raw-tps-bench raw_tps_bench_test.c
	@mkdir -p $(dir $(RAW_TPS_TEST_BIN))
	$(CC) $(CFLAGS) -o $(RAW_TPS_TEST_BIN) raw_tps_bench_test.c $(LDLIBS)
	$(RAW_TPS_TEST_RUN)

clean:
	rm -f oxidize-c test_oc oc-merge raw-tps-bench raw-tps-bench-test

.PHONY: merge test raw-tps-bench raw-tps-bench-test clean

CUDA_HOME ?= /usr/local/cuda
NVCC ?= nvcc
CUDA_GENCODE ?= -gencode arch=compute_90,code=sm_90
CUDA_BIN = $(if $(BIN),$(BIN),oxidize-c-cuda)
CUDA_OBJ = $(CUDA_BIN).cuda.o
CUDA_TEST_BIN = $(if $(TEST_BIN),$(TEST_BIN),cuda-test)
BATCH_SMOKE_BIN = $(if $(BIN),$(BIN),batch-smoke)
CUDA_COMMON_SRC = $(filter-out main.c,$(SRC))

cuda: $(SRC) oc.h $(CUDA_OBJ)
	@mkdir -p $(dir $(CUDA_BIN))
	$(CC) $(CFLAGS) -DOC_CUDA -o $(CUDA_BIN) $(SRC) $(CUDA_OBJ) -L$(CUDA_HOME)/lib64 -lcublas -lcudart -lstdc++ $(LDLIBS)

$(CUDA_OBJ): cuda.cu oc.h
	@mkdir -p $(dir $@)
	$(NVCC) -O3 $(CUDA_GENCODE) -c $< -o $@

cuda-test: cuda_test.cu cuda.cu oc.h
	@if test ! -f cuda_test.cu; then \
		echo "error: cuda_test.cu is required for cuda-test; add the CUDA test source first" >&2; \
		exit 2; \
	fi
	@mkdir -p $(dir $(CUDA_TEST_BIN))
	$(CC) $(CFLAGS) -c quant.c -o $(CUDA_TEST_BIN).quant.o
	$(NVCC) -O3 $(CUDA_GENCODE) -o $(CUDA_TEST_BIN) cuda_test.cu cuda.cu $(CUDA_TEST_BIN).quant.o -L$(CUDA_HOME)/lib64 -lcublas -lcudart
	$(CUDA_TEST_BIN)

batch-smoke: batch_smoke.c $(CUDA_COMMON_SRC) oc.h $(CUDA_OBJ)
	@mkdir -p $(dir $(BATCH_SMOKE_BIN))
	$(CC) $(CFLAGS) -DOC_CUDA -o $(BATCH_SMOKE_BIN) batch_smoke.c $(CUDA_COMMON_SRC) $(CUDA_OBJ) -L$(CUDA_HOME)/lib64 -lcublas -lcudart -lstdc++ $(LDLIBS)

.PHONY: cuda cuda-test batch-smoke
