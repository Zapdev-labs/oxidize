# ============================================================================
# UNVERIFIED — this Makefile and everything it builds have NEVER been compiled
# or run in the authoring environment. Requires macOS + Xcode Command Line Tools
# (`xcrun metal`, `xcrun metallib`, clang++ with Metal/Foundation). It MAY NOT
# COMPILE. Not part of the default oxidize-c build.
# ============================================================================
#
# Opt-in Metal backend build. From this directory:
#   make
#   make clean
#
# Produces:
#   build/oxidize.metallib   — gk_* + lk_* kernels
#   build/*.o                — host ObjC++ objects (link from a macOS binary)
#
# Point the runtime at the metallib via $OXIDIZE_METAL_LIB or the default path
# src/metal/build/oxidize.metallib (relative to the process cwd).

METAL    ?= xcrun -sdk macosx metal
METALLIB ?= xcrun -sdk macosx metallib
CXX      ?= clang++
BUILD    := build

CXXFLAGS := -std=c++17 -fobjc-arc -O2 -Wall -Wextra \
            -I. -I.. \
            -fno-objc-arc-exceptions
LDFLAGS  := -framework Metal -framework Foundation

AIR := $(BUILD)/gemma4.air $(BUILD)/llama.air
LIB := $(BUILD)/oxidize.metallib
OBJS := $(BUILD)/metal_common.o $(BUILD)/gemma4_metal.o $(BUILD)/llama_metal.o

.PHONY: all clean info

all: $(LIB) $(OBJS)
	@echo "metal: built $(LIB) and host objects (UNVERIFIED — never run here)"

info:
	@echo "UNVERIFIED Metal backend"
	@echo "  metallib: $(LIB)"
	@echo "  objects:  $(OBJS)"
	@echo "  link with: $(LDFLAGS)"
	@echo "  runtime:  OXIDIZE_METAL_LIB=$(LIB)  (or default path)"

$(BUILD):
	mkdir -p $(BUILD)

# Compile MSL -> AIR (include metal_dequant.h from this directory).
$(BUILD)/gemma4.air: gemma4.metal metal_dequant.h | $(BUILD)
	$(METAL) -c -I. gemma4.metal -o $@

$(BUILD)/llama.air: llama.metal metal_dequant.h | $(BUILD)
	$(METAL) -c -I. llama.metal -o $@

$(LIB): $(AIR)
	$(METALLIB) $(AIR) -o $@

$(BUILD)/metal_common.o: metal_common.mm metal_common.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c metal_common.mm -o $@

$(BUILD)/gemma4_metal.o: gemma4_metal.mm gemma4_metal.h metal_common.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c gemma4_metal.mm -o $@

$(BUILD)/llama_metal.o: llama_metal.mm llama_metal.h metal_common.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c llama_metal.mm -o $@

clean:
	rm -rf $(BUILD)
