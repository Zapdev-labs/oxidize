# Go OpenAI-compatible server over the oxidize-c C ABI. Depends on the static
# liboxidize.a in the parent dir; this target builds it via the C Makefile's
# `lib` target (it does not edit that Makefile). `make -C server server`.
OXROOT := $(abspath ..)
LIB    := $(OXROOT)/liboxidize.a

.PHONY: server test clean
server: $(LIB)
	go build -o oxidize-serve .

test: $(LIB)
	go test ./...

$(LIB):
	$(MAKE) -C $(OXROOT) lib

clean:
	rm -f oxidize-serve
