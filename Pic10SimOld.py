import random


class Pic10Sim:
    def __init__(self):
        # --- CONSTANTS (PIC10F200 Specs) ---
        self.RAM_SIZE = 32  # 0x00 to 0x1F (includes SFRs + RAM)
        self.FLASH_SIZE = 256  # 256 words of program memory

        # --- STATE REGISTERS ---
        self.ram = bytearray(self.RAM_SIZE)
        self.w_reg = 0
        self.pc = 0
        self.stack = []

        # --- STORAGE ---
        self.program = []  # The Genome
        self.opcode_map = {}  # Integer -> String Mapping

        # --- SPECIAL REGISTERS ---
        self.tris_gpio = 0xFF
        self.option_reg = 0xFF

        # --- METRICS ---
        self.cycles = 0
        self.crash_reason = None

    def load(self, program, opcode_list):
        """
        Loads the genome and the opcode translation table.
        """
        self.program = program
        self.opcode_map = opcode_list  # List of strings ['ADDWF_W', etc]
        self.reset()

    def reset(self, fixed_tris=None):
        self.ram = bytearray(self.RAM_SIZE)
        self.w_reg = 0
        self.pc = 0xFF  # Power-on Reset Vector
        self.stack = []
        self.cycles = 0
        self.crash_reason = None
        self.ram[0x03] = 0x18  # Status POR value

        if fixed_tris is not None:
            self.tris_gpio = fixed_tris
        else:
            self.tris_gpio = 0xFF  # Default Input

    def check_stack_for_macro(self):
        if len(self.stack) >= 2:
            self.crash_reason = "STACK_OVERFLOW_ON_DELAY"
            return False
        return True

    def emulate_cycle(self, gp3_input):
        """
        The Heart of the Thunderdome.
        1. Injects Cosmic Signal (GP3).
        2. Fetches & Decodes Instruction from self.program.
        3. Executes Step.
        4. Returns Output.
        """
        if self.crash_reason: return 0  # Dead organism

        # --- 1. UPDATE HARDWARE (Inject Signal) ---
        # GP3 is Input-Only. Force RAM bit to match Cosmic Input.
        if gp3_input:
            self.ram[0x06] |= (1 << 3)
        else:
            self.ram[0x06] &= ~(1 << 3)

        # --- 2. FETCH ---
        # Handle PC wrapping and Empty Programs
        if not self.program: return 0

        # PC is 9-bit, but Flash is 256 words. Wrap to length.
        # Note: Real PICs wrap at 0xFF to 0x00.
        fetch_pc = self.pc & 0xFF
        if fetch_pc >= len(self.program):
            fetch_pc %= len(self.program)  # Safety wrap for short genomes

        instruction = self.program[fetch_pc]
        op_id = instruction[0]
        operand = instruction[1]

        # Translate Opcode ID to String Mnemonic
        if 0 <= op_id < len(self.opcode_map):
            mnemonic = self.opcode_map[op_id]
        else:
            mnemonic = 'NOP'  # Safety fallback

        # --- 3. DECODE (Bit Extraction) ---
        # For Bit Instructions (BCF, BSF, BTFSC, BTFSS),
        # we split the 8-bit Operand: [BBB AAAAA]
        # Top 3 bits = Bit Index (0-7), Bottom 5 bits = Address.
        bit_index = 0
        if mnemonic in ['BCF', 'BSF', 'BTFSC', 'BTFSS']:
            bit_index = (operand >> 5) & 0x07
            operand = operand & 0x1F  # Mask address

        # --- 4. EXECUTE ---
        self.step(mnemonic, operand, bit_index)

        # --- 5. READ OUTPUT ---
        gpio_val = self.ram[0x06]
        return (gpio_val & 0x03)

    def step(self, action_type, operand, bit_index=0):
        cycle_cost = 1
        pc_increment = 1

        # --- MACROS ---
        if action_type == 'DELAY_MACRO':
            if not self.check_stack_for_macro(): return True
            # Formula: Delay = (Operand^3)*0.1 + 10
            cycles_burned = int((operand ** 3) * 0.1) + 10
            self.cycles += cycles_burned
            self.pc = (self.pc + 1) & 0xFF
            return False

        # --- CONTROL FLOW ---
        if action_type == 'GOTO':
            self.cycles += 2
            self.pc = operand & 0xFF
            return False

        elif action_type == 'CALL':
            if len(self.stack) >= 2:
                self.crash_reason = "STACK_OVERFLOW_HARDWARE"
                return True
            self.stack.append((self.pc + 1) & 0xFF)
            self.cycles += 2
            self.pc = operand & 0xFF
            return False

        elif action_type == 'RETLW':
            if len(self.stack) == 0:
                self.crash_reason = "STACK_UNDERFLOW"
                return True
            return_addr = self.stack.pop()
            self.w_reg = operand & 0xFF
            self.cycles += 2
            self.pc = return_addr
            return False

        # --- BIT MANIPULATION ---
        elif action_type == 'BCF':
            val = self.read_ram(operand)
            mask = ~(1 << bit_index) & 0xFF
            self.write_ram(operand, val & mask)

        elif action_type == 'BSF':
            val = self.read_ram(operand)
            mask = (1 << bit_index) & 0xFF
            self.write_ram(operand, val | mask)

        elif action_type == 'BTFSC':
            val = self.read_ram(operand)
            if (val & (1 << bit_index)) == 0:
                self.pc = (self.pc + 1) & 0xFF
                self.cycles += 1

        elif action_type == 'BTFSS':
            val = self.read_ram(operand)
            if (val & (1 << bit_index)) != 0:
                self.pc = (self.pc + 1) & 0xFF
                self.cycles += 1

        # --- SKIP ON ZERO ---
        elif action_type.startswith('DECFSZ'):
            val = self.read_ram(operand)
            res = (val - 1) & 0xFF
            if action_type == 'DECFSZ_W': self.w_reg = res
            if action_type == 'DECFSZ_F': self.write_ram(operand, res)
            if res == 0:
                self.pc = (self.pc + 1) & 0xFF
                self.cycles += 1

        elif action_type.startswith('INCFSZ'):
            val = self.read_ram(operand)
            res = (val + 1) & 0xFF
            if action_type == 'INCFSZ_W': self.w_reg = res
            if action_type == 'INCFSZ_F': self.write_ram(operand, res)
            if res == 0:
                self.pc = (self.pc + 1) & 0xFF
                self.cycles += 1

        # --- ALU ---
        elif action_type.startswith('MOVF'):
            val = self.read_ram(operand)
            self.update_flags(0, 0, val, 'LOGIC')
            if action_type == 'MOVF_W': self.w_reg = val
            if action_type == 'MOVF_F': self.write_ram(operand, val)

        elif action_type.startswith('ADDWF'):
            val = self.read_ram(operand)
            res = self.w_reg + val
            self.update_flags(self.w_reg, val, res, 'ADD')
            if action_type == 'ADDWF_W': self.w_reg = res & 0xFF
            if action_type == 'ADDWF_F': self.write_ram(operand, res)

        elif action_type.startswith('SUBWF'):
            val = self.read_ram(operand)
            res = val - self.w_reg
            self.update_flags(val, self.w_reg, res, 'SUB')
            if action_type == 'SUBWF_W': self.w_reg = res & 0xFF
            if action_type == 'SUBWF_F': self.write_ram(operand, res)

        elif action_type.startswith('ANDWF'):
            val = self.read_ram(operand)
            res = self.w_reg & val
            self.update_flags(0, 0, res, 'LOGIC')
            if action_type == 'ANDWF_W': self.w_reg = res
            if action_type == 'ANDWF_F': self.write_ram(operand, res)

        elif action_type.startswith('IORWF'):
            val = self.read_ram(operand)
            res = self.w_reg | val
            self.update_flags(0, 0, res, 'LOGIC')
            if action_type == 'IORWF_W': self.w_reg = res
            if action_type == 'IORWF_F': self.write_ram(operand, res)

        elif action_type.startswith('XORWF'):
            val = self.read_ram(operand)
            res = self.w_reg ^ val
            self.update_flags(0, 0, res, 'LOGIC')
            if action_type == 'XORWF_W': self.w_reg = res
            if action_type == 'XORWF_F': self.write_ram(operand, res)

        elif action_type.startswith('COMF'):
            val = self.read_ram(operand)
            res = (~val) & 0xFF
            self.update_flags(0, 0, res, 'LOGIC')
            if action_type == 'COMF_W': self.w_reg = res
            if action_type == 'COMF_F': self.write_ram(operand, res)

        elif action_type.startswith('INCF'):
            val = self.read_ram(operand)
            res = (val + 1) & 0xFF
            self.update_flags(0, 0, res, 'LOGIC')
            if action_type == 'INCF_W': self.w_reg = res
            if action_type == 'INCF_F': self.write_ram(operand, res)

        elif action_type.startswith('DECF'):
            val = self.read_ram(operand)
            res = (val - 1) & 0xFF
            self.update_flags(0, 0, res, 'LOGIC')
            if action_type == 'DECF_W': self.w_reg = res
            if action_type == 'DECF_F': self.write_ram(operand, res)

        elif action_type == 'SWAPF':
            val = self.read_ram(operand)
            res = ((val & 0x0F) << 4) | ((val & 0xF0) >> 4)
            if action_type == 'SWAPF_W': self.w_reg = res
            if action_type == 'SWAPF_F': self.write_ram(operand, res)

        elif action_type.startswith('RLF'):
            val = self.read_ram(operand)
            carry_in = (self.ram[0x03] & 0x01)
            carry_out = (val & 0x80) >> 7
            res = ((val << 1) & 0xFF) | carry_in
            if carry_out:
                self.ram[0x03] |= 1
            else:
                self.ram[0x03] &= ~1
            if action_type == 'RLF_W': self.w_reg = res
            if action_type == 'RLF_F': self.write_ram(operand, res)

        elif action_type.startswith('RRF'):
            val = self.read_ram(operand)
            carry_in = (self.ram[0x03] & 0x01)
            carry_out = (val & 0x01)
            res = (val >> 1) | (carry_in << 7)
            if carry_out:
                self.ram[0x03] |= 1
            else:
                self.ram[0x03] &= ~1
            if action_type == 'RRF_W': self.w_reg = res
            if action_type == 'RRF_F': self.write_ram(operand, res)

        elif action_type == 'CLRW':
            self.w_reg = 0
            self.update_flags(0, 0, 0, 'LOGIC')

        elif action_type == 'CLRF':
            self.write_ram(operand, 0)
            self.update_flags(0, 0, 0, 'LOGIC')

        elif action_type == 'MOVWF':
            self.write_ram(operand, self.w_reg)

        elif action_type == 'MOVLW':
            self.w_reg = operand & 0xFF

        # --- UPDATE PC/CYCLES ---
        self.cycles += cycle_cost
        self.pc = (self.pc + pc_increment) & 0xFF
        if self.cycles > 2000000:
            self.crash_reason = "TIMEOUT"
            return True
        return False

    def read_ram(self, f):
        address = f & 0x1F
        if address == 0x00: address = self.ram[0x04] & 0x1F
        return self.ram[address]

    def write_ram(self, f, value):
        address = f & 0x1F
        value = value & 0xFF
        if address == 0x00: address = self.ram[0x04] & 0x1F

        if address == 0x02:
            self.pc = value
        elif address == 0x03:
            mask = 0b00011000
            self.ram[0x03] = (value & ~mask) | (self.ram[0x03] & mask)
        elif address == 0x06:
            # PROTECT INPUT PIN GP3
            current = self.ram[0x06]
            # Write only to GP0, GP1, GP2 (Mask 0xF7 clears bit 3)
            # Restore GP3 from current state
            self.ram[0x06] = (value & 0xF7) | (current & 0x08)
        else:
            self.ram[address] = value

    def update_flags(self, val1, val2, result, operation):
        status = self.ram[0x03]
        if (result & 0xFF) == 0:
            status |= 4
        else:
            status &= ~4

        if operation in ['ADD', 'SUB']:
            if operation == 'ADD':
                if result > 255:
                    status |= 1
                else:
                    status &= ~1
                if ((val1 & 0xF) + (val2 & 0xF)) > 0xF:
                    status |= 2
                else:
                    status &= ~2
            elif operation == 'SUB':
                if result >= 0:
                    status |= 1
                else:
                    status &= ~1
                if ((val1 & 0xF) - (val2 & 0xF)) >= 0:
                    status |= 2
                else:
                    status &= ~2

        self.ram[0x03] = status
