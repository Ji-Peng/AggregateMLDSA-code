# natver: pq-crystals ML-DSA-44 AVX2 batch verification.
#
# The ISA is pinned to AVX2 and BMI2 so that the binary contains no AVX-512
# instruction even when built on a machine with those extensions. Never add
# -march=native: it would emit EVEX encodings on an AVX-512 host.

CC ?= gcc
DILITHIUM ?= ../vendor/dilithium
DIR := $(DILITHIUM)/avx2

CFLAGS := -O3 -mavx2 -mbmi2 -mpopcnt -maes -fopenmp -DDILITHIUM_MODE=2 \
          -I$(DIR) -Wall -Wextra
LDFLAGS := -fopenmp

SOURCES := sign.c packing.c polyvec.c poly.c consts.c rejsample.c rounding.c \
           fips202.c fips202x4.c symmetric-shake.c
ASM := ntt.S invntt.S pointwise.S shuffle.S f1600x4.S
OBJS := $(addprefix build/,$(SOURCES:.c=.o)) $(addprefix build/,$(ASM:.S=.o))

.PHONY: all clean

all: natver

natver: natver.c randombytes.c sha256.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ natver.c randombytes.c sha256.c $(OBJS) $(LDFLAGS)

build/%.o: $(DIR)/%.c
	@mkdir -p build
	$(CC) $(CFLAGS) -c -o $@ $<

build/%.o: $(DIR)/%.S
	@mkdir -p build
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf build natver
