import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys

# Correctly import the simulator class
try:
    from Pic10Sim import Pic10Sim
except ImportError:
    print("CRITICAL ERROR: Pic10Sim.py not found in the current directory.")
    sys.exit()

# The exact opcode list from your Driver files
OPCODE_LIST = [
    'ADDWF_W', 'ADDWF_F', 'ANDWF_W', 'ANDWF_F',
    'CLRF', 'CLRW', 'COMF_W', 'COMF_F',
    'DECF_W', 'DECF_F', 'DECFSZ_W', 'DECFSZ_F',
    'INCF_W', 'INCF_F', 'INCFSZ_W', 'INCFSZ_F',
    'IORWF_W', 'IORWF_F', 'MOVF_W', 'MOVF_F',
    'MOVWF', 'MOVLW', 'NOP', 'RLF_W', 'RLF_F',
    'RRF_W', 'RRF_F', 'SUBWF_W', 'SUBWF_F',
    'SWAPF_W', 'SWAPF_F', 'XORWF_W', 'XORWF_F',
    'BCF', 'BSF', 'BTFSC', 'BTFSS',
    'CALL', 'GOTO', 'RETLW', 'DELAY_MACRO'
]


# -------------------------------------------------------------------------
# 1. The Diagnostic Cosmos Generator
# -------------------------------------------------------------------------
def generate_diagnostic_cosmos(total_ticks=600):
    """
    Generates a deterministic 600-tick stimulus array to probe organism behavior.
    """
    inputs = []

    HUM = [1, 1, 1, 1, 0, 0, 0, 0]
    DUET = [1, 1, 0, 0, 1, 1, 0, 0]
    WARBLE = [1, 0, 1, 0, 1, 0, 1, 0]

    def add_block(signal, inverted=False, gap=10):
        seq = [1 - x if inverted else x for x in signal]
        inputs.extend(seq)
        inputs.extend([0] * gap)

    inputs.extend([0] * 50)  # Block 0: Squelch Test

    add_block(HUM, inverted=False)
    add_block(HUM, inverted=True)

    add_block(DUET, inverted=False)
    add_block(DUET, inverted=True)

    add_block(WARBLE, inverted=False)
    add_block(WARBLE, inverted=True)

    random.seed(42)
    signals = [HUM, DUET, WARBLE]

    while len(inputs) < total_ticks:
        sig = random.choice(signals)
        invert = random.choice([True, False])
        gap = random.randint(4, 20)

        seq = [1 - x if invert else x for x in sig]
        inputs.extend(seq)
        inputs.extend([0] * gap)

    return np.array(inputs[:total_ticks])


# -------------------------------------------------------------------------
# 2. Evaluation Engine (Corrected for Pic10Sim API)
# -------------------------------------------------------------------------
def evaluate_critter(genome, stimulus):
    """
    Runs a single genome through the Pic10Sim virtual core.
    Correctly accounts for multi-tick instructions (dt).
    """
    ticks = len(stimulus)
    outputs = []

    sim = Pic10Sim()
    sim.reset()
    sim.load(genome, OPCODE_LIST)

    t = 0
    while t < ticks:
        bit0 = stimulus[t]

        # Execute exactly one instruction
        out, did_crash = sim.emulate_cycle(gp2_input=0, gp3_input=bit0)

        if did_crash:
            outputs.extend([0] * (ticks - t))
            break

        # How much world-time elapsed during this instruction?
        dt = getattr(sim, "last_dt", 1)
        if not isinstance(dt, int) or dt < 1:
            dt = 1

        if t + dt > ticks:
            dt = ticks - t

        # The output of GP0 is bit 0 of the `out` variable
        gp0_val = out & 1

        # Hold the output constant for dt ticks
        outputs.extend([gp0_val] * dt)

        t += dt

    return np.array(outputs[:ticks])


# -------------------------------------------------------------------------
# 3. Tagging & Math Engine
# -------------------------------------------------------------------------
def profile_critter(stimulus, output):
    profile = {
        'is_noisy': False,
        'is_repeater': False,
        'is_inverter': False,
        'is_lowpass': False,
        'best_lag': None,
        'max_corr': 0.0
    }

    if np.var(output[0:50]) > 0:
        profile['is_noisy'] = True

    if np.var(stimulus) > 0 and np.var(output) > 0:
        norm_stim = (stimulus - np.mean(stimulus)) / np.std(stimulus)
        norm_out = (output - np.mean(output)) / np.std(output)

        corr = np.correlate(norm_out, norm_stim, mode='full') / len(stimulus)
        zero_lag_idx = len(stimulus) - 1

        max_c = 0.0
        best_l = 0
        for lag in range(0, 16):
            c = corr[zero_lag_idx - lag]
            if abs(c) > abs(max_c):
                max_c = c
                best_l = lag

        profile['max_corr'] = max_c
        profile['best_lag'] = best_l

        if max_c > 0.80:
            profile['is_repeater'] = True
        elif max_c < -0.80:
            profile['is_inverter'] = True

    hum_var = np.var(output[50:90])
    warble_var = np.var(output[120:160])

    if hum_var > 0.1 and warble_var < 0.05:
        profile['is_lowpass'] = True

    return profile


# -------------------------------------------------------------------------
# 4. Plotting & Visualization
# -------------------------------------------------------------------------
def plot_population_heatmap(stimulus, all_outputs, profiles, out_filename):
    def sort_key(item):
        idx, prof = item
        score = 0
        if prof['is_repeater']:
            score = 100 + prof['max_corr']
        elif prof['is_inverter']:
            score = 50 + abs(prof['max_corr'])
        elif prof['is_lowpass']:
            score = 25
        elif prof['is_noisy']:
            score = 10
        elif prof['max_corr'] == 0:
            score = -1
        else:
            score = 5
        return score

    sorted_items = sorted(enumerate(profiles), key=sort_key, reverse=True)
    sorted_indices = [i[0] for i in sorted_items]

    sorted_outputs = np.array(all_outputs)[sorted_indices]

    fig, (ax_stim, ax_pop) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 8]})

    ax_stim.step(range(len(stimulus)), stimulus, where='post', color='black', linewidth=1.5)
    ax_stim.set_title("Standard Diagnostic Cosmos (GP3 Input)", fontsize=14)
    ax_stim.set_xlim(0, len(stimulus))
    ax_stim.set_yticks([0, 1])
    ax_stim.grid(True, alpha=0.3)

    ax_pop.imshow(sorted_outputs, aspect='auto', cmap='binary', interpolation='none')
    ax_pop.set_title("Population Output Matrix (GP0)", fontsize=14)
    ax_pop.set_ylabel("Organisms (Sorted by Behavioral Archetype)", fontsize=12)
    ax_pop.set_xlabel("Simulation Clock Ticks", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_filename, dpi=150)
    plt.close()
    print(f"Saved population heatmap to {out_filename}")


def plot_oscilloscope(stimulus, output, bin_key, out_filename):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].step(range(len(stimulus)), stimulus, where='post', color='blue')
    axes[0].set_ylabel("GP3 (Ear/Cosmos)")
    axes[0].set_yticks([0, 1])
    axes[0].grid(True, alpha=0.3)

    axes[1].step(range(len(output)), output, where='post', color='red')
    axes[1].set_ylabel("GP0 (Mouth)")
    axes[1].set_yticks([0, 1])
    axes[1].grid(True, alpha=0.3)

    plt.xlabel("Clock Ticks")
    plt.suptitle(f"Logic Trace for Organism {bin_key}")
    plt.tight_layout()
    plt.savefig(out_filename)
    plt.close()


# -------------------------------------------------------------------------
# 5. Main Execution
# -------------------------------------------------------------------------
def main(archive_path):
    print(f"Loading archive from {archive_path}...")
    open_func = gzip.open if archive_path.endswith('.gz') else open

    with open_func(archive_path, 'rt') as f:
        data = json.load(f)

    archive = data.get('archive', {})
    if not archive:
        print("Error: Could not find 'archive' key in JSON.")
        return

    print(f"Loaded {len(archive)} organisms. Generating Cosmos...")
    stimulus = generate_diagnostic_cosmos(600)

    all_outputs = []
    profiles = []
    bin_keys = list(archive.keys())

    print("Running Lockstep Evaluation & Tagging...")
    for i, bin_key in enumerate(bin_keys):
        # Depending on if it's Phase 1 vs Phase 4 formatting, safely extract genome
        genome = archive[bin_key].get('genome', [])

        output = evaluate_critter(genome, stimulus)
        all_outputs.append(output)

        prof = profile_critter(stimulus, output)
        profiles.append(prof)

        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{len(bin_keys)} critters...")

    print("Generating Heatmap...")
    out_dir = "Time_Dynamics_Results"
    os.makedirs(out_dir, exist_ok=True)

    heatmap_file = os.path.join(out_dir, "Population_Heatmap_Phase1_2.png")
    plot_population_heatmap(stimulus, all_outputs, profiles, heatmap_file)

    sample_key = bin_keys[0]
    plot_oscilloscope(stimulus, all_outputs[0], sample_key, os.path.join(out_dir, "Trace_Sample.png"))

    print("Done!")


if __name__ == "__main__":
    target_archive = "Phase2_Results/run1/backup_phase2_40000000.json.gz"
    main(target_archive)