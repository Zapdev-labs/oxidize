CXX ?= g++
AR ?= ar
CXXFLAGS ?= -O3 -fPIC -std=c++17 -Wall -Wextra -Iinclude
LDFLAGS ?= -shared

UNAME_M := $(shell uname -m 2>/dev/null || echo unknown)
ifeq ($(UNAME_M),x86_64)
  CXXFLAGS += -mavx2 -mfma
endif
ifeq ($(UNAME_M),amd64)
  CXXFLAGS += -mavx2 -mfma
endif

SRCS := src/q4k_scalar.cpp src/q8k.cpp src/q4k_avx2.cpp src/cpu_detect.cpp \
        src/dot_f32.cpp src/gemv.cpp src/oxk_api.cpp
OBJS := $(SRCS:.cpp=.o)

PREFIX ?= /usr/local
LIBDIR := $(abspath .)

.PHONY: all clean install static test

all: liboxk.so liboxk.a

test: liboxk.so
	$(CXX) $(CXXFLAGS) -o test/smoke test/smoke.cpp -L. -loxk -Wl,-rpath,.
	./test/smoke

liboxk.so: $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

liboxk.a: $(OBJS)
	$(AR) rcs $@ $^

static: liboxk.a

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

install: liboxk.so
	install -d $(PREFIX)/lib $(PREFIX)/include
	install -m 644 liboxk.so $(PREFIX)/lib/
	install -m 644 include/oxk.h $(PREFIX)/include/

clean:
	rm -f $(OBJS) liboxk.so liboxk.a
