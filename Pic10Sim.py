class Pic10Sim:
    """
    PIC10F200-inspired emulator for MAP-Elites evolutionary experiments.

    ISA CONTRACT (LOCKED):
      - Split destination opcodes (_W vs _F)
      - Fixed GPIO directions:
          GP3, GP2 = input-only
          GP1, GP0 = output-only
      - Reset vector fixed at PC = 0
    """

    # -------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------

    def __init__(self):
        self.RAM_SIZE = 32
        self.FLASH_SIZE = 256

        # GPIO masks
        self.GPIO_OUT_MASK = 0b00000011   # GP0, GP1
        self.GPIO_IN_MASK  = 0b00001100   # GP2, GP3
        self.GPIO_KEEP_MASK = 0xFF ^ self.GPIO_OUT_MASK

        self.reset()

    def reset(self):
        self.ram = bytearray(self.RAM_SIZE)
        self.w_reg = 0
        self.pc = 0
        self.stack = []
        self.cycles = 0
        self.crash_reason = None

        # STATUS register (stable POR state)
        self.ram[0x03] = 0x18

        # GPIO initial state
        self.ram[0x06] = 0x00

    def load(self, program, opcode_list):
        self.program = program
        self.opcode_list = opcode_list

    # -------------------------------------------------
    # MEMORY ACCESS
    # -------------------------------------------------

    def read_ram(self, addr):
        addr &= 0x1F
        if addr == 0x00:  # INDF
            fsr = self.ram[0x04] & 0x1F
            return self.ram[fsr]
        return self.ram[addr]

    def write_ram(self, addr, value):
        addr &= 0x1F
        value &= 0xFF

        if addr == 0x00:  # INDF
            fsr = self.ram[0x04] & 0x1F
            self.ram[fsr] = value
            return

        if addr == 0x06:  # GPIO
            old = self.ram[0x06]
            new = (old & self.GPIO_KEEP_MASK) | (value & self.GPIO_OUT_MASK)
            self.ram[0x06] = new
            return

        self.ram[addr] = value

    # -------------------------------------------------
    # EXECUTION
    # -------------------------------------------------

    def emulate_cycle(self, gp2_input=0, gp3_input=0):
        if self.crash_reason:
            return 0, True

        # --- inject inputs ---
        gpio = self.ram[0x06]
        gpio = (gpio | (1 << 2)) if gp2_input else (gpio & ~(1 << 2))
        gpio = (gpio | (1 << 3)) if gp3_input else (gpio & ~(1 << 3))
        self.ram[0x06] = gpio

        # --- fetch ---
        if not self.program:
            return 0, False

        idx = self.pc % len(self.program)
        op_id, operand = self.program[idx]
        opcode = self.opcode_list[op_id]

        # --- decode bit ops ---
        bit = None
        if opcode in ("BCF", "BSF", "BTFSC", "BTFSS"):
            bit = (operand >> 5) & 7
            operand &= 0x1F

        # --- execute ---
        self.step(opcode, operand, bit)

        return self.ram[0x06] & self.GPIO_OUT_MASK, bool(self.crash_reason)

    # -------------------------------------------------
    # INSTRUCTION SEMANTICS
    # -------------------------------------------------

    def step(self, op, f, bit=None):
        if self.crash_reason:
            return

        self.cycles += 1

        # -------- CONTROL FLOW --------

        if op == "NOP":
            self.pc += 1
            return

        if op == "GOTO":
            self.pc = f
            self.cycles += 1
            return

        if op == "CALL":
            if len(self.stack) >= 2:
                self.crash_reason = "Stack overflow"
                return
            self.stack.append(self.pc + 1)
            self.pc = f
            self.cycles += 1
            return

        if op == "RETLW":
            if not self.stack:
                self.crash_reason = "Stack underflow"
                return
            self.w_reg = f
            self.pc = self.stack.pop()
            self.cycles += 1
            return

        if op == "DELAY_MACRO":
            self.cycles += int((f ** 3) * 0.1) + 10
            self.pc += 1
            return

        # -------- BIT OPS --------

        if op == "BCF":
            self.write_ram(f, self.read_ram(f) & ~(1 << bit))
            self.pc += 1
            return

        if op == "BSF":
            self.write_ram(f, self.read_ram(f) | (1 << bit))
            self.pc += 1
            return

        if op == "BTFSC":
            self.pc += 2 if not (self.read_ram(f) & (1 << bit)) else 1
            self.cycles += 1
            return

        if op == "BTFSS":
            self.pc += 2 if (self.read_ram(f) & (1 << bit)) else 1
            self.cycles += 1
            return

        # -------- REGISTER / ALU OPS --------

        val = self.read_ram(f)

        def set_z(x):
            if x == 0:
                self.ram[0x03] |= 0x04
            else:
                self.ram[0x03] &= ~0x04

        if op == "MOVLW":
            self.w_reg = f
            set_z(self.w_reg)
            self.pc += 1
            return

        if op == "MOVWF":
            self.write_ram(f, self.w_reg)
            self.pc += 1
            return

        if op == "MOVF_W":
            self.w_reg = val
            set_z(val)
            self.pc += 1
            return

        if op == "MOVF_F":
            set_z(val)
            self.pc += 1
            return

        # ---- arithmetic helpers ----

        def alu(result, write_w):
            result &= 0xFF
            set_z(result)
            if write_w:
                self.w_reg = result
            else:
                self.write_ram(f, result)

        if op == "ADDWF_W":
            alu(self.w_reg + val, True)
        elif op == "ADDWF_F":
            alu(self.w_reg + val, False)

        elif op == "SUBWF_W":
            alu(val - self.w_reg, True)
        elif op == "SUBWF_F":
            alu(val - self.w_reg, False)

        elif op == "ANDWF_W":
            alu(self.w_reg & val, True)
        elif op == "ANDWF_F":
            alu(self.w_reg & val, False)

        elif op == "IORWF_W":
            alu(self.w_reg | val, True)
        elif op == "IORWF_F":
            alu(self.w_reg | val, False)

        elif op == "XORWF_W":
            alu(self.w_reg ^ val, True)
        elif op == "XORWF_F":
            alu(self.w_reg ^ val, False)

        elif op == "COMF_W":
            alu(~val, True)
        elif op == "COMF_F":
            alu(~val, False)

        elif op == "INCF_W":
            alu(val + 1, True)
        elif op == "INCF_F":
            alu(val + 1, False)

        elif op == "DECF_W":
            alu(val - 1, True)
        elif op == "DECF_F":
            alu(val - 1, False)

        elif op == "SWAPF_W":
            alu(((val & 0x0F) << 4) | ((val & 0xF0) >> 4), True)
        elif op == "SWAPF_F":
            alu(((val & 0x0F) << 4) | ((val & 0xF0) >> 4), False)

        elif op == "RLF_W":
            alu(((val << 1) | (self.ram[0x03] & 1)), True)
        elif op == "RLF_F":
            alu(((val << 1) | (self.ram[0x03] & 1)), False)

        elif op == "RRF_W":
            alu((val >> 1), True)
        elif op == "RRF_F":
            alu((val >> 1), False)

        elif op == "INCFSZ_W":
            result = (val + 1) & 0xFF
            self.w_reg = result
            self.pc += 2 if result == 0 else 1
            return

        elif op == "INCFSZ_F":
            result = (val + 1) & 0xFF
            self.write_ram(f, result)
            self.pc += 2 if result == 0 else 1
            return

        elif op == "DECFSZ_W":
            result = (val - 1) & 0xFF
            self.w_reg = result
            self.pc += 2 if result == 0 else 1
            return

        elif op == "DECFSZ_F":
            result = (val - 1) & 0xFF
            self.write_ram(f, result)
            self.pc += 2 if result == 0 else 1
            return

        else:
            # Unknown opcode = NOP
            self.pc += 1
            return

        self.pc += 1
