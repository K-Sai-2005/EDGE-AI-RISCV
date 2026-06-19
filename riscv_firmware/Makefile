# Makefile — Crack Detection Edge AI Firmware
# RISC-V bare-metal build for SiFive FE310-G002
# ──────────────────────────────────────────────────────────────────────

# Toolchain (install: https://github.com/sifive/freedom-tools/releases)
CROSS   = riscv64-unknown-elf
CC      = $(CROSS)-gcc
OBJCOPY = $(CROSS)-objcopy
SIZE    = $(CROSS)-size

# Target architecture: RV32IMAC (FE310-G002)
ARCH    = rv32imac
ABI     = ilp32
CFLAGS  = -march=$(ARCH) -mabi=$(ABI) -mcmodel=medany \
          -O2 -Wall -Wextra -ffunction-sections -fdata-sections \
          -DSIMULATION_MODE   # ← Remove this flag for real board build

LDFLAGS = -march=$(ARCH) -mabi=$(ABI) \
          -Wl,--gc-sections \
          -T link.ld          # FE310 linker script (from Freedom SDK)

# Source files
SRCS = main.c \
       inference_engine.c \
       uart_utils.c \
       model_params.c

TARGET = crack_detect
ELF    = $(TARGET).elf
BIN    = $(TARGET).bin
HEX    = $(TARGET).hex

# ── Build targets ─────────────────────────────────────────────────────
all: $(ELF)
	@echo ""
	@echo "=== Build complete ==="
	$(SIZE) $(ELF)

$(ELF): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $@

$(BIN): $(ELF)
	$(OBJCOPY) -O binary $< $@

$(HEX): $(ELF)
	$(OBJCOPY) -O ihex $< $@

# ── Simulation target (Freedom Studio GDB semihosting) ───────────────
simulate: $(ELF)
	@echo "Open Freedom Studio and load $(ELF) in the debug config."
	@echo "Set SIMULATION_MODE=1 in launch config preprocessor defines."

# ── Flash to board (requires OpenOCD + VSDSquadron config) ────────────
flash: $(ELF)
	openocd -f interface/ftdi/vsdsquadron.cfg \
	        -f target/sifive.cfg \
	        -c "program $(ELF) verify reset exit"

clean:
	rm -f $(ELF) $(BIN) $(HEX) *.o

.PHONY: all simulate flash clean
