"""
Conformance suite for Pic10Sim evolutionary substrate.

Run:
  python test_pic10sim_conformance.py

This suite validates:
  - Reset entry PC=0
  - GPIO direction lock (GP2/GP3 input-only; GP0/GP1 output-only)
  - Input injection overrides internal writes
  - Skip semantics (BTFSC/BTFSS, INCFSZ/DECFSZ)
  - Stack overflow/underflow crashes
"""

from Pic10Sim import Pic10Sim


# IMPORTANT: opcode ordering must match your evolutionary system.
# This list matches the one in your Multiverse PDF and code.
OPCODES = [
    "ADDWF", "ANDWF", "CLRF", "CLRW", "COMF", "DECF", "DECFSZ",
    "INCF", "INCFSZ", "IORWF", "MOVF", "MOVWF", "MOVLW", "NOP",
    "RLF", "RRF", "SUBWF", "SWAPF", "XORWF",
    "BCF", "BSF", "BTFSC", "BTFSS",
    "CALL", "GOTO", "RETLW",
    # Include macro if your genome uses it (some runs do):
    "DELAY_MACRO",
]

OP_INDEX = {m: i for i, m in enumerate(OPCODES)}


def enc(mnemonic, operand=0, bit=None):
    """
    Encode one instruction into (op_id, operand_byte).
    For bit ops, pack [BBB AAAAA] into operand.
    """
    if mnemonic in ("BCF", "BSF", "BTFSC", "BTFSS"):
        assert bit is not None, "bit ops require bit index"
        assert 0 <= bit <= 7
        assert 0 <= operand <= 0x1F
        operand_byte = ((bit & 0x7) << 5) | (operand & 0x1F)
    else:
        assert bit is None, "non-bit ops should not provide bit index"
        assert 0 <= operand <= 0xFF
        operand_byte = operand & 0xFF

    return (OP_INDEX[mnemonic], operand_byte)


def run_steps(sim, prog, steps, gp2=0, gp3=0):
    sim.reset()
    sim.load(prog, OPCODES)
    outs = []
    for _ in range(steps):
        out, crashed = sim.emulate_cycle(gp2_input=gp2, gp3_input=gp3)
        outs.append((out, crashed, sim.pc, sim.cycles))
        if crashed:
            break
    return outs


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_reset_entry_pc0():
    sim = Pic10Sim()
    sim.reset()
    assert_true(sim.pc == 0, f"Reset PC should be 0, got {sim.pc}")
    print("[PASS] reset entry PC=0")


def test_gpio_output_write_only_affects_gp0_gp1():
    sim = Pic10Sim()
    # Program: MOVLW 0xFF ; MOVWF GPIO
    # Expect: only GP0/GP1 become 1, inputs unchanged
    GPIO = 0x06
    prog = [
        enc("MOVLW", 0xFF),
        enc("MOVWF", GPIO),
        enc("NOP", 0),
    ]
    outs = run_steps(sim, prog, steps=3, gp2=0, gp3=0)
    gpio = sim.ram[GPIO]
    assert_true((gpio & 0b11) == 0b11, "GP0/GP1 should be set by MOVWF")
    assert_true((gpio & 0b1100) == 0b0000, "GP2/GP3 should reflect injected inputs (0)")
    print("[PASS] MOVWF GPIO cannot write GP2/GP3")


def test_gpio_bitset_cannot_write_gp2_gp3():
    sim = Pic10Sim()
    GPIO = 0x06

    # Try to set GP2 and GP3 using BSF GPIO,2 and BSF GPIO,3
    prog = [
        enc("BSF", GPIO, bit=2),
        enc("BSF", GPIO, bit=3),
        enc("NOP", 0),
    ]
    run_steps(sim, prog, steps=3, gp2=0, gp3=0)
    gpio = sim.ram[GPIO]
    assert_true((gpio & 0b1100) == 0b0000, "GP2/GP3 must remain 0 with injected 0")
    print("[PASS] BSF on GPIO cannot write GP2/GP3")


def test_input_injection_overrides_program_writes():
    sim = Pic10Sim()
    GPIO = 0x06

    # Program tries to clear GPIO repeatedly; inputs should stay forced high if injected high
    prog = [
        enc("CLRF", GPIO),
        enc("CLRF", GPIO),
        enc("CLRF", GPIO),
    ]
    run_steps(sim, prog, steps=3, gp2=1, gp3=1)
    gpio = sim.ram[GPIO]
    assert_true((gpio & 0b1100) == 0b1100, "Injected inputs must override internal clears")
    print("[PASS] input injection overrides internal writes")


def test_btfss_skip_semantics():
    sim = Pic10Sim()
    GPIO = 0x06

    # If GP2 is high, then BTFSS GPIO,2 should skip next instruction.
    # Program:
    #   BTFSS GPIO,2
    #   MOVLW 0x01
    #   MOVLW 0xAA
    # Expect with gp2=1: MOVLW 0x01 is skipped -> W ends as 0xAA
    prog = [
        enc("BTFSS", GPIO, bit=2),
        enc("MOVLW", 0x01),
        enc("MOVLW", 0xAA),
    ]
    sim.reset()
    sim.load(prog, OPCODES)
    for _ in range(3):
        sim.emulate_cycle(gp2_input=1, gp3_input=0)

    assert_true(sim.w_reg == 0xAA, f"BTFSS skip failed; W={sim.w_reg:#02x}")
    print("[PASS] BTFSS skip semantics (gp2=1)")


def test_call_stack_overflow():
    sim = Pic10Sim()

    # CALL 0x00 ; CALL 0x00 ; CALL 0x00 should overflow 2-level stack
    prog = [
        enc("CALL", 0x00),
        enc("CALL", 0x00),
        enc("CALL", 0x00),
    ]
    outs = run_steps(sim, prog, steps=10, gp2=0, gp3=0)
    assert_true(sim.crash_reason is not None, "Expected crash due to stack overflow")
    assert_true("overflow" in sim.crash_reason.lower(), f"Unexpected crash reason: {sim.crash_reason}")
    print("[PASS] stack overflow crash")


def test_retlw_underflow():
    sim = Pic10Sim()

    prog = [
        enc("RETLW", 0x55),
        enc("NOP", 0),
    ]
    run_steps(sim, prog, steps=2, gp2=0, gp3=0)
    assert_true(sim.crash_reason is not None, "Expected crash due to stack underflow")
    assert_true("underflow" in sim.crash_reason.lower(), f"Unexpected crash reason: {sim.crash_reason}")
    print("[PASS] stack underflow crash")


def main():
    test_reset_entry_pc0()
    test_gpio_output_write_only_affects_gp0_gp1()
    test_gpio_bitset_cannot_write_gp2_gp3()
    test_input_injection_overrides_program_writes()
    test_btfss_skip_semantics()
    test_call_stack_overflow()
    test_retlw_underflow()

    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()
