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
# 1. Dual Diagnostic Cosmos Generator
# -------------------------------------------------------------------------
def generate_diagnostic_cosmos_dual(total_ticks=800):
    target = []
    stim_A = []
    stim_B = []

    HUM = [1, 1, 1, 1, 0, 0, 0, 0]
    DUET = [1, 1, 0, 0, 1, 1, 0, 0]
    WARBLE = [1, 0, 1, 0, 1, 0, 1, 0]

    def add_block(signal, inverted=False, gap=10, blind_A=False, blind_B=False):
        seq = [1 - x if inverted else x for x in signal]
        target.extend(seq)
        stim_A.extend([0 if blind_A else x for x in seq])
        stim_B.extend([0 if blind_B else x for x in seq])

        target.extend([0] * gap)
        stim_A.extend([0] * gap)
        stim_B.extend([0] * gap)

    # Block 0: Squelch Test
    target.extend([0] * 50);
    stim_A.extend([0] * 50);
    stim_B.extend([0] * 50)

    # Block 1, 2, 3: Standard Frequencies
    add_block(HUM, inverted=False);
    add_block(HUM, inverted=True)
    add_block(DUET, inverted=False);
    add_block(DUET, inverted=True)
    add_block(WARBLE, inverted=False);
    add_block(WARBLE, inverted=True)

    # Block 4: The Intermittent Truth (Dropout) Probe (Ticks 158 to ~210)
    add_block(HUM * 2, inverted=False, blind_A=True)
    add_block(HUM * 2, inverted=False, blind_B=True)

    # Block 5: Ecological Gauntlet
    random.seed(42)
    signals = [HUM, DUET, WARBLE]

    while len(target) < total_ticks:
        sig = random.choice(signals)
        invert = random.choice([True, False])
        gap = random.randint(4, 20)

        blind_A = random.random() < 0.10
        blind_B = random.random() < 0.10 if not blind_A else False

        add_block(sig, inverted=invert, gap=gap, blind_A=blind_A, blind_B=blind_B)

    return (np.array(target[:total_ticks]),
            np.array(stim_A[:total_ticks]),
            np.array(stim_B[:total_ticks]))


# -------------------------------------------------------------------------
# 2. Dual-Core Lockstep Evaluation Engine
# -------------------------------------------------------------------------
def evaluate_pair(genome_A, genome_B, stim_A, stim_B):
    ticks = len(stim_A)
    outputs_A = []
    outputs_B = []

    core_A = Pic10Sim()
    core_B = Pic10Sim()
    core_A.reset();
    core_B.reset()
    core_A.load(genome_A, OPCODE_LIST)
    core_B.load(genome_B, OPCODE_LIST)

    t_A = 0;
    t_B = 0

    def read_pin(history, current_time):
        if current_time <= 0 or not history: return 0
        idx = current_time - 1
        return history[idx] if idx < len(history) else history[-1]

    while t_A < ticks or t_B < ticks:
        if t_A <= t_B and t_A < ticks:
            b3_A = stim_A[t_A]
            b2_A = read_pin(outputs_B, t_A)
            out_A, crash_A = core_A.emulate_cycle(gp2_input=b2_A, gp3_input=b3_A)
            if crash_A:
                outputs_A.extend([0] * (ticks - t_A))
                t_A = ticks
            else:
                dt_A = max(1, getattr(core_A, "last_dt", 1))
                if t_A + dt_A > ticks: dt_A = ticks - t_A
                gp0_A = out_A & 1
                outputs_A.extend([gp0_A] * dt_A)
                t_A += dt_A

        if t_B < t_A and t_B < ticks:
            b3_B = stim_B[t_B]
            b2_B = read_pin(outputs_A, t_B)
            out_B, crash_B = core_B.emulate_cycle(gp2_input=b2_B, gp3_input=b3_B)
            if crash_B:
                outputs_B.extend([0] * (ticks - t_B))
                t_B = ticks
            else:
                dt_B = max(1, getattr(core_B, "last_dt", 1))
                if t_B + dt_B > ticks: dt_B = ticks - t_B
                gp0_B = out_B & 1
                outputs_B.extend([gp0_B] * dt_B)
                t_B += dt_B

    return np.array(outputs_A[:ticks]), np.array(outputs_B[:ticks])


# -------------------------------------------------------------------------
# 3. Tagging & Math Engine
# -------------------------------------------------------------------------
def profile_pair(target, out_A, out_B):
    profile = {
        'is_noisy': False,
        'is_repeater': False,
        'is_inverter': False,
        'is_lowpass': False,
        'max_corr': 0.0,
        'pair_sync': 0.0
    }

    if np.var(out_A[0:50]) > 0: profile['is_noisy'] = True

    if np.var(target) > 0 and np.var(out_A) > 0:
        norm_targ = (target - np.mean(target)) / np.std(target)
        norm_A = (out_A - np.mean(out_A)) / np.std(out_A)
        corr = np.correlate(norm_A, norm_targ, mode='full') / len(target)
        zero_lag_idx = len(target) - 1

        max_c = 0.0
        for lag in range(0, 16):
            c = corr[zero_lag_idx - lag]
            if abs(c) > abs(max_c): max_c = c
        profile['max_corr'] = max_c

        if max_c > 0.80:
            profile['is_repeater'] = True
        elif max_c < -0.80:
            profile['is_inverter'] = True

    if np.var(out_A) > 0 and np.var(out_B) > 0:
        norm_A = (out_A - np.mean(out_A)) / np.std(out_A)
        norm_B = (out_B - np.mean(out_B)) / np.std(out_B)
        sync_corr = np.correlate(norm_A, norm_B, mode='valid')[0] / len(target)
        profile['pair_sync'] = sync_corr

    hum_var = np.var(out_A[50:90])
    warble_var = np.var(out_A[120:160])
    if hum_var > 0.1 and warble_var < 0.05: profile['is_lowpass'] = True

    return profile


# -------------------------------------------------------------------------
# 4. Plotting & Visualization
# -------------------------------------------------------------------------
def plot_population_heatmap(target, all_outputs_A, profiles, out_filename, world_id):
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
    sorted_outputs = np.array(all_outputs_A)[sorted_indices]

    fig, (ax_stim, ax_pop) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 8]})

    ax_stim.step(range(len(target)), target, where='post', color='black', linewidth=1.5)
    ax_stim.set_title("Absolute Target Cosmos", fontsize=14)
    ax_stim.set_xlim(0, len(target))
    ax_stim.set_yticks([0, 1])
    ax_stim.grid(True, alpha=0.3)

    ax_pop.imshow(sorted_outputs, aspect='auto', cmap='binary', interpolation='none')
    ax_pop.set_title(f"Phase 4 ({world_id}) Population Matrix: Alice GP0", fontsize=14)
    ax_pop.set_ylabel("Organism Pairs (Sorted by Behavior)", fontsize=12)
    ax_pop.set_xlabel("Simulation Clock Ticks", fontsize=12)

    # Corrected Dropout Zone shading (Ticks 158 to 210)
    ax_pop.axvspan(158, 210, color='red', alpha=0.15, label="Dropout Zone")
    ax_pop.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_filename, dpi=150)
    plt.close()
    print(f"Saved population heatmap to {out_filename}")


def plot_dual_oscilloscope(target, stim_A, stim_B, out_A, out_B, bin_key, out_filename, world_id):
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

    axes[0].step(range(len(target)), target, where='post', color='black')
    axes[0].set_ylabel("Absolute Target")

    axes[1].step(range(len(stim_A)), stim_A, where='post', color='blue')
    axes[1].set_ylabel("Alice GP3 (In)")

    axes[2].step(range(len(stim_B)), stim_B, where='post', color='cyan')
    axes[2].set_ylabel("Bob GP3 (In)")

    axes[3].step(range(len(out_A)), out_A, where='post', color='red')
    axes[3].set_ylabel("Alice GP0 (Out)")

    axes[4].step(range(len(out_B)), out_B, where='post', color='magenta')
    axes[4].set_ylabel("Bob GP0 (Out)")

    for ax in axes:
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)
        ax.axvspan(158, 210, color='red', alpha=0.1)  # Corrected Dropout Zone

    plt.xlabel("Clock Ticks")
    plt.suptitle(f"Phase 4 ({world_id}) Logic Trace for Pair {bin_key}")
    plt.tight_layout()
    plt.savefig(out_filename)
    plt.close()


# -------------------------------------------------------------------------
# 5. Main Execution Loop for a Specific World
# -------------------------------------------------------------------------
def run_world_analysis(world_id, archive_path):
    print(f"\n{'=' * 50}\nStarting Analysis for World: {world_id}\n{'=' * 50}")

    if not os.path.exists(archive_path):
        print(f"File not found: {archive_path}. Skipping...")
        return

    open_func = gzip.open if archive_path.endswith('.gz') else open
    with open_func(archive_path, 'rt') as f:
        data = json.load(f)

    archive_A = data.get('archive', {}).get('alice', {})
    archive_B = data.get('archive', {}).get('bob', {})

    if not archive_A or not archive_B:
        print(f"Error: Missing 'alice' or 'bob' archives in {world_id} JSON.")
        return

    # Only test pairs that exist in both archives at the exact same niche coordinate
    common_bins = list(set(archive_A.keys()).intersection(set(archive_B.keys())))
    print(f"Loaded {len(common_bins)} valid pairs for {world_id}. Generating Dual Cosmos...")

    target, stim_A, stim_B = generate_diagnostic_cosmos_dual(800)

    all_outputs_A = []
    all_outputs_B = []
    profiles = []

    print("Running Lockstep Evaluation & Tagging...")
    for i, bin_key in enumerate(common_bins):
        genome_A = archive_A[bin_key].get('genome', [])
        genome_B = archive_B[bin_key].get('genome', [])

        out_A, out_B = evaluate_pair(genome_A, genome_B, stim_A, stim_B)

        all_outputs_A.append(out_A)
        all_outputs_B.append(out_B)

        prof = profile_pair(target, out_A, out_B)
        profiles.append(prof)

        if i > 0 and i % 500 == 0:
            print(f"  Processed {i}/{len(common_bins)} pairs...")

    print(f"Generating Output Plots for {world_id}...")
    out_dir = f"Time_Dynamics_Phase4_{world_id}_Results"
    os.makedirs(out_dir, exist_ok=True)

    heatmap_file = os.path.join(out_dir, f"{world_id}_Population_Heatmap.png")
    plot_population_heatmap(target, all_outputs_A, profiles, heatmap_file, world_id)

    if profiles:
        best_idx = np.argmax([p['max_corr'] for p in profiles])
        best_key = common_bins[best_idx]
        plot_dual_oscilloscope(target, stim_A, stim_B, all_outputs_A[best_idx], all_outputs_B[best_idx],
                               best_key, os.path.join(out_dir, f"{world_id}_Trace_Best_Pair.png"), world_id)

    print(f"Finished World {world_id}!")


if __name__ == "__main__":
    # IMPORTANT: Update this base path to point to the folder containing your 5 Phase 4 JSON files.
    # Replace "Phase4_Results" with your actual directory path.
    base_dir = "Phase4_Results/run1"

    # Run the 5 worlds sequentially
    run_world_analysis("I0", f"{base_dir}/I0_backup_phase4_002000000.json.gz")
    run_world_analysis("I1", f"{base_dir}/I1_backup_phase4_002000000.json.gz")
    run_world_analysis("I2", f"{base_dir}/I2_backup_phase4_002000000.json.gz")
    run_world_analysis("I3", f"{base_dir}/I3_backup_phase4_002000000.json.gz")
    run_world_analysis("I4", f"{base_dir}/I4_backup_phase4_002000000.json.gz")