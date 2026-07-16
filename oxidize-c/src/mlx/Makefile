# ============================================================================
# UNVERIFIED — NEVER COMPILED OR RUN.
# Written blind against src/cuda/ + oxidize-core/src/backends/mlx.rs.
# Requires a Mac with Apple Silicon + mlx-c. MAY NOT COMPILE.
# ----------------------------------------------------------------------------
# Opt-in MLX backend objects. NEVER linked into the default oxidize-c
# `make` / `make test` path. Build here, then manually link the resulting
# .o files into a custom binary if/when wiring the CLI.
#
# Dependencies (install on the Mac; paths vary by how you built mlx-c):
#
#   MLX_INCLUDE  — directory containing <mlx/c/mlx.h>
#                  e.g.  /usr/local/include   or  $(brew --prefix)/include
#   MLX_LIB      — directory containing libmlxc / libmlx (names ASSUMED)
#                  e.g.  /usr/local/lib       or  $(brew --prefix)/lib
#
# Example:
#   make -C src/mlx MLX_INCLUDE=/opt/homebrew/include MLX_LIB=/opt/homebrew/lib
#
# The parent Makefile must NOT include this directory. Validators wire the
# objects by hand when bringing up the Mac path.
# ============================================================================

MLX_INCLUDE ?= /usr/local/include
MLX_LIB     ?= /usr/local/lib

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra
CFLAGS  += -I.. -I$(MLX_INCLUDE)
# Parent src/ symbols (gguf, quant, model_*) are NOT linked here — these are
# object files only. A full binary also needs the rest of oxidize-c + mlx-c.
LDFLAGS ?= -L$(MLX_LIB) -lmlxc -lmlx -lc++

SRCS = mlx_common.c gemma4_mlx.c llama_mlx.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean print-flags

all: $(OBJS)
	@echo "mlx: built $(OBJS) (UNVERIFIED — not linked into oxidize-c)"
	@echo "mlx: link manually with: \$$(CC) ... $(OBJS) -L$(MLX_LIB) -lmlxc -lmlx ..."

print-flags:
	@echo "MLX_INCLUDE=$(MLX_INCLUDE)"
	@echo "MLX_LIB=$(MLX_LIB)"
	@echo "CFLAGS=$(CFLAGS)"
	@echo "LDFLAGS=$(LDFLAGS)"

%.o: %.c mlx_common.h mlx_backend.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS)

# Documented (not automated) full-link sketch for a validator on Mac:
#   $(CC) -o oxidize-c-mlx $(OXIDIZE_OBJS) $(OBJS) $(LDFLAGS)
# where OXIDIZE_OBJS is everything the main binary already builds, plus a
# CLI branch that calls gemma4_mlx_init / llama_mlx_init. That branch does
# NOT exist yet and must not be added under the default Makefile.
