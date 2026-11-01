# natver: pq-crystals ML-DSA AVX2 batch verification, one binary per parameter set:
# natver (ML-DSA-44, DILITHIUM_MODE=2), natver65 (mode 3) and natver87 (mode 5).
#
# The ISA is pinned to AVX2 and BMI2 so that the binaries contain no AVX-512
# instruction even when built on a machine with those extensions. Never add
# -march=native: it would emit EVEX encodings on an AVX-512 host.

CC ?= gcc
DILITHIUM ?= ../vendor/dilithium
DIR := $(DILITHIUM)/avx2

CFLAGS := -O3 -mavx2 -mbmi2 -mpopcnt -maes -fopenmp -I$(DIR) -Wall -Wextra
LDFLAGS := -fopenmp

SOURCES := sign.c packing.c polyvec.c poly.c consts.c rejsample.c rounding.c \
           fips202.c fips202x4.c symmetric-shake.c
ASM := ntt.S invntt.S pointwise.S shuffle.S f1600x4.S

# Binary name per DILITHIUM_MODE.
NAME_2 := natver
NAME_3 := natver65
NAME_5 := natver87
MODES := 2 3 5

.PHONY: all clean

all: $(foreach m,$(MODES),$(NAME_$(m)))

# Objects of mode $(1) live in build$(1)/.
define mode_rules
OBJS_$(1) := $$(addprefix build$(1)/,$$(SOURCES:.c=.o)) $$(addprefix build$(1)/,$$(ASM:.S=.o))

$$(NAME_$(1)): natver.c randombytes.c sha256.c $$(OBJS_$(1))
	$$(CC) $$(CFLAGS) -DDILITHIUM_MODE=$(1) -o $$@ natver.c randombytes.c sha256.c $$(OBJS_$(1)) $$(LDFLAGS)

build$(1)/%.o: $$(DIR)/%.c
	@mkdir -p build$(1)
	$$(CC) $$(CFLAGS) -DDILITHIUM_MODE=$(1) -c -o $$@ $$<

build$(1)/%.o: $$(DIR)/%.S
	@mkdir -p build$(1)
	$$(CC) $$(CFLAGS) -DDILITHIUM_MODE=$(1) -c -o $$@ $$<
endef

$(foreach m,$(MODES),$(eval $(call mode_rules,$(m))))

clean:
	rm -rf build2 build3 build5 $(foreach m,$(MODES),$(NAME_$(m)))
