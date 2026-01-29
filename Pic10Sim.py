import random


class Pic10Sim:
    """
    PIC10F200-inspired emulator, tuned for evolutionary experiments.

    HARDWARE CONTRACT (LOCKED):
      - GPIO directions are fixed (TRIS removed from evolving instruction set):
          GP3, GP2: INPUT-ONLY
          GP1, GP0: OUTPUT-ONLY
      - External inputs are injected each cycle and overwrite GPIO input bits.
      - Programs start at reset entry PC=0 (hardware-faithful for deployment).
    """

    def __init__(self):
        # --- CONSTANTS (PIC10F200-like Specs) ---
        self.RAM_SIZE = 32          # 0x00 to 0x1F (includes SFRs + RAM)
        self.FLASH_SIZE = 256       # 256 words program memory

        # --- GPIO DIRECTION (FIXED) ---
        # TRIS bit = 1 => input, 0 => output (PIC convention)
        # GP3 (bit 3) input, GP2 (bit 2) input, GP1 (bit 1) output, GP0 (bit 0) output
        self.FIXED_TRIS = 0b00001100

        # Convenience masks
        self.GPIO_OUT_MASK = 0b00000011  # GP0, GP1
        self.GPIO_IN_MASK  = 0b00001100  # GP2, GP3
        self.GPIO_KEEP_MASK = 0xFF ^ self.GPIO_OUT_MASK  # keep all non-output bits on GPIO writes

        # --- STATE REGISTERS ---
        self.ram = bytearray(self.RAM_SIZE)
        self.w_reg = 0
        self.pc = 0
        self.stack = []

        # --- STORAGE ---
        self.program = []         # list of (op_id, operand)
        self.opcode_map = []      # list of mnemonics indexed by op_id

        # --- RUN STATE ---
        self.cycles = 0
        self.crash_reason = None

        self.reset()

    def reset(self, fixed_tris=None):
        """
        Reset to a known power-on state, with a hardware-faithful reset entry point.
        """
        self.ram = bytearray(self.RAM_SIZE)
        self.w_reg = 0
        self.pc = 0               # *** LOCKED RESET ENTRY (PC=0) ***
        self.stack = []
        self.cycles = 0
        self.crash_reason = None

        # PIC-like POR status value (you had 0x18; keep it stable)
        self.ram[0x03] = 0x18

        # TRIS is fixed for this project, but allow override for experiments.
        if fixed_tris is None:
            self.tris_gpio = self.FIXED_TRIS
        else:
            self.tris_gpio = fixed_tris & 0xFF

        # Initialize GPIO to 0 (outputs low, inputs will be injected)
        self.ram[0x06] = 0x00

    def load(self, program, opcode_list):
        """
        Loads the genome and opcode translation table.
        program: list[(op_id:int, operand:int)]
        opcode_list: list[str] mapping op_id -> mnemonic
        """
        self.program = program
        self.opcode_map = opcode_list

    # -------------------------
    # RAM ACCESS HELPERS
    # -------------------------
    def read_ram(self, addr):
        """
        Read RAM/SFR. Implements INDF indirection via FSR at 0x04 when addr==0.
        """
        addr &= 0x1F

        # INDF / indirect addressing
        if addr == 0x00:
            fsr = self.ram[0x04] & 0x1F
            return self.ram[fsr]

        return self.ram[addr]

    def write_ram(self, addr, value):
        """
        Write RAM/SFR. Enforces fixed GPIO directions when writing GPIO (0x06).
        Implements INDF indirection via FSR at 0x04 when addr==0.
        """
        addr &= 0x1F
        value &= 0xFF

        # INDF / indirect addressing
        if addr == 0x00:
            fsr = self.ram[0x04] & 0x1F
            self.ram[fsr] = value
            return

        # GPIO direction lock: GP2/GP3 are input-only, GP0/GP1 output-only
        if addr == 0x06:
            old = self.ram[0x06]
            # Preserve input bits (and any other non-output bits) on write;
            # Only allow GP0/GP1 to be changed by program writes.
            new_val = (old & self.GPIO_KEEP_MASK) | (value & self.GPIO_OUT_MASK)
            self.ram[0x06] = new_val
            return

        # Default write
        self.ram[addr] = value

    # -------------------------
    # EXECUTION CORE
    # -------------------------
    def check_stack_for_macro(self):
        # Macro consumes stack depth; keep consistent with your existing crash semantics
        if len(self.stack) >= 2:
            self.crash_reason = "Stack overflow (Macro)"
            return False
        return True

    def emulate_cycle(self, gp2_input=0, gp3_input=0):
        """
        Canonical single-cycle interface:
          1) Injects external inputs (GP2, GP3) into GPIO input bits.
          2) Fetch + decode instruction at PC.
          3) Execute one instruction.
          4) Return outputs (GP1:GP0) as 2-bit integer plus crash status.

        Returns: (out2bit:int, crashed:bool)
        """
        if self.crash_reason:
            return 0, True

        # --- 1) INJECT EXTERNAL INPUTS (force input pins) ---
        gpio = self.ram[0x06]

        # Force GP2
        if gp2_input:
            gpio |= (1 << 2)
        else:
            gpio &= ~(1 << 2)

        # Force GP3
        if gp3_input:
            gpio |= (1 << 3)
        else:
            gpio &= ~(1 << 3)

        # Apply back, but keep outputs unchanged (outputs already included in gpio)
        self.ram[0x06] = gpio

        # --- 2) FETCH ---
        if not self.program:
            return 0, False

        fetch_pc = self.pc & 0xFF
        if fetch_pc >= len(self.program):
            # Hardware-faithful wrap for full 256-word programs would be 0xFF->0x00.
            # For shorter genomes, keep your safety behavior (wrap to length).
            fetch_pc %= len(self.program)

        op_id, operand = self.program[fetch_pc]
        operand &= 0xFF

        if 0 <= op_id < len(self.opcode_map):
            mnemonic = self.opcode_map[op_id]
        else:
            mnemonic = "NOP"

        # --- 3) DECODE BIT OPS (BBB AAAAA) ---
        bit_index = 0
        if mnemonic in ["BCF", "BSF", "BTFSC", "BTFSS"]:
            bit_index = (operand >> 5) & 0x07
            operand = operand & 0x1F

        # --- 4) EXECUTE ---
        self.step(mnemonic, operand, bit_index)

        # --- 5) READ OUTPUTS ---
        out2 = self.ram[0x06] & self.GPIO_OUT_MASK
        return out2, bool(self.crash_reason)

    def step(self, action_type, operand, bit_index=0):
        """
        Execute a single decoded instruction.
        Returns True if crashed, else False.
        """
        if self.crash_reason:
            return True

        # Timeout guard (keeps evolution from hanging forever)
        if self.cycles > 200000:
            self.crash_reason = "Timeout"
            return True

        # Default costs
        cycle_cost = 1
        pc_increment = 1

        # --- MACRO ---
        if action_type == "DELAY_MACRO":
            if not self.check_stack_for_macro():
                return True
            cycles_burned = int((operand ** 3) * 0.1) + 10
            self.cycles += cycles_burned
            self.pc = (self.pc + 1) & 0xFF
            return False

        # --- CONTROL FLOW ---
        if action_type == "GOTO":
            self.cycles += 2
            self.pc = operand & 0xFF
            return False

        if action_type == "CALL":
            self.cycles += 2
            if len(self.stack) >= 2:
                self.crash_reason = "Stack overflow (CALL)"
                return True
            self.stack.append((self.pc + 1) & 0xFF)
            self.pc = operand & 0xFF
            return False

        if action_type == "RETLW":
            self.cycles += 2
            self.w_reg = operand & 0xFF
            if not self.stack:
                self.crash_reason = "Stack underflow (RETLW)"
                return True
            self.pc = self.stack.pop()
            return False

        # --- BIT OPS ---
        if action_type == "BCF":
            val = self.read_ram(operand)
            val &= ~(1 << bit_index)
            self.write_ram(operand, val)
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "BSF":
            val = self.read_ram(operand)
            val |= (1 << bit_index)
            self.write_ram(operand, val)
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "BTFSC":
            val = self.read_ram(operand)
            if (val & (1 << bit_index)) == 0:
                pc_increment = 2
                cycle_cost = 2
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "BTFSS":
            val = self.read_ram(operand)
            if (val & (1 << bit_index)) != 0:
                pc_increment = 2
                cycle_cost = 2
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        # --- REGISTER OPS / ALU ---
        # NOTE: This is intentionally minimal and PIC-flavored for your instruction subset.
        # It preserves your existing evolutionary semantics rather than full silicon fidelity.

        if action_type == "NOP":
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "CLRW":
            self.w_reg = 0
            self.ram[0x03] |= 0x04  # Z
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "CLRF":
            self.write_ram(operand, 0)
            self.ram[0x03] |= 0x04  # Z
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "MOVLW":
            self.w_reg = operand & 0xFF
            # Z flag if W becomes 0
            if self.w_reg == 0:
                self.ram[0x03] |= 0x04
            else:
                self.ram[0x03] &= ~0x04
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "MOVWF":
            self.write_ram(operand, self.w_reg)
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "MOVF":
            val = self.read_ram(operand)
            self.w_reg = val
            if val == 0:
                self.ram[0x03] |= 0x04
            else:
                self.ram[0x03] &= ~0x04
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "INCF":
            val = (self.read_ram(operand) + 1) & 0xFF
            self.write_ram(operand, val)
            if val == 0:
                self.ram[0x03] |= 0x04
            else:
                self.ram[0x03] &= ~0x04
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "DECF":
            val = (self.read_ram(operand) - 1) & 0xFF
            self.write_ram(operand, val)
            if val == 0:
                self.ram[0x03] |= 0x04
            else:
                self.ram[0x03] &= ~0x04
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "INCFSZ":
            val = (self.read_ram(operand) + 1) & 0xFF
            self.write_ram(operand, val)
            if val == 0:
                pc_increment = 2
                cycle_cost = 2
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        if action_type == "DECFSZ":
            val = (self.read_ram(operand) - 1) & 0xFF
            self.write_ram(operand, val)
            if val == 0:
                pc_increment = 2
                cycle_cost = 2
            self.cycles += cycle_cost
            self.pc = (self.pc + pc_increment) & 0xFF
            return False

        # Fallback: treat unknown as NOP (keeps evolution stable)
        self.cycles += cycle_cost
        self.pc = (self.pc + pc_increment) & 0xFF
        return False
