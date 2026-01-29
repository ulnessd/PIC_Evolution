import random
import concurrent.futures
import os
import json
import gzip
import zlib
import time
import sys

# ==============================================================================
# CONFIGURATION: PHASE 2 (GENERATIONAL PULSE)
# ==============================================================================
# PHYSICS SETTINGS ("Grandmaster" Difficulty)
PHYSICS_CYCLES = 15000
NOISE_RATE = 0.04

# GENERATIONAL SETTINGS
ROUNDS = 25  # How many "Pulses" to run
PAIRS_PER_ROUND = 50000  # Number of mating pairs
CHILDREN_PER_PAIR = 5  # Children per pair (Total = 250,000 new orgs per round)
MUTATION_RATE = 0.05  # Low mutation (we want recombination, not noise)

# HARDWARE
NUM_WORKERS = max(1, os.cpu_count() - 2)

# SYSTEM
GRID_BINS = [16, 16, 16, 16]
MAX_LINES = 256
SOURCE_BATCH = 5000  # Which batch from Phase 1 to load

# PIC10SIM
try:
    from Pic10Sim import Pic10Sim
except ImportError:
    print("CRITICAL ERROR: Pic10Sim.py not found.")
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


# ==============================================================================
# PHYSICS ENGINE
# ==============================================================================
class Cosmos:
    def __init__(self, length):
        self.length = length
        self.inputs = []
        self.targets = []
        self._generate()

    def _generate(self):
        PATTERNS = {
            1: [1, 1, 1, 1, 0, 0, 0, 0],
            2: [1, 1, 0, 0, 1, 1, 0, 0],
            3: [1, 0, 1, 0, 1, 0, 1, 0],
            0: [0, 0, 0, 0, 0, 0, 0, 0]
        }
        t = 0
        rts_state = 0
        while t < self.length:
            target = random.choice([0, 1, 2, 3])
            pattern = PATTERNS[target]
            invert = (random.random() < 0.5)
            num_packets = random.randint(15, 25)
            for _ in range(num_packets):
                for bit in pattern:
                    if t >= self.length: break
                    val = (1 - bit) if invert else bit
                    if random.random() < NOISE_RATE:
                        rts_state = 1 - rts_state
                    val = val ^ rts_state
                    self.inputs.append(val)
                    self.targets.append(target)
                    t += 1
            gap = random.randint(4, 20)
            for _ in range(gap):
                if t >= self.length: break
                self.inputs.append(0)
                self.targets.append(0)
                t += 1


def evaluate_organism(genome):
    sim = Pic10Sim()
    sim.load(genome, OPCODE_LIST)
    cosmos = Cosmos(PHYSICS_CYCLES)

    score = 0
    valid_cycles = 0
    grace = 0
    last_target = 0
    last_out = 0
    toggles = 0

    decoded_prog = []
    for instr in genome:
        op_id, raw_operand = instr
        if op_id >= len(OPCODE_LIST): op_id = 22
        mnemonic = OPCODE_LIST[op_id]
        bit_index = (raw_operand >> 5) & 0x07
        operand = raw_operand & 0x1F
        decoded_prog.append((mnemonic, operand, bit_index))

    for t in range(PHYSICS_CYCLES):
        inp = cosmos.inputs[t]
        target = cosmos.targets[t]
        current_gpio = sim.ram[0x06]
        if inp:
            sim.ram[0x06] = (current_gpio & ~0x04) | 0x04
        else:
            sim.ram[0x06] = (current_gpio & ~0x04)

        pc = sim.pc
        if pc >= len(genome): pc %= len(genome)
        mnemonic, op, bit = decoded_prog[pc]
        sim.step(mnemonic, op, bit)
        if sim.crash_reason: break

        out = sim.ram[0x06] & 0x03
        if out != last_out: toggles += 1
        last_out = out

        if target != last_target: grace = 16
        last_target = target

        if grace > 0:
            grace -= 1
        else:
            valid_cycles += 1
            if out == target: score += 1

    fitness = score / max(1, valid_cycles)

    metrics = {}
    if len(genome) > 0:
        metrics = {
            "Activity": toggles / PHYSICS_CYCLES,
            "Responsiveness": fitness,
            "ChannelEntropy": 0.0,
            "AlgoDensity": len(genome) / MAX_LINES
        }
    return genome, fitness, metrics


def worker(chunk):
    results = []
    for genome in chunk:
        try:
            results.append(evaluate_organism(genome))
        except:
            results.append((genome, 0.0, {}))
    return results


# ==============================================================================
# GENETICS
# ==============================================================================
def mutate(genome):
    ng = [list(i) for i in genome]
    for i in range(len(ng)):
        if random.random() < MUTATION_RATE:
            if random.random() < 0.5:
                ng[i][0] = random.randint(0, len(OPCODE_LIST) - 1)
            else:
                ng[i][1] = random.randint(0, 255)
    if random.random() < (MUTATION_RATE * 0.5):
        if len(ng) < MAX_LINES and random.random() < 0.5:
            ng.insert(random.randint(0, len(ng)), [random.randint(0, 40), random.randint(0, 255)])
        elif len(ng) > 1:
            ng.pop(random.randint(0, len(ng) - 1))
    return ng


def crossover(p1, p2):
    # Rule 0: Safety Check (If parents are empty, return empty)
    if len(p1) == 0: return list(p2)
    if len(p2) == 0: return list(p1)

    roll = random.random()

    # --- MECHANISM 1: HOMOLOGOUS CROSSOVER (85%) ---
    # Standard "Cut and Splice" to preserve functional blocks
    if roll < 0.85:
        min_len = min(len(p1), len(p2))
        if min_len < 2: return list(p1)  # Too short to split
        pt = random.randint(1, min_len - 1)
        return p1[:pt] + p2[pt:]

    # --- MECHANISM 2: TRANSPOSON INSERTION (12%) ---
    # Horizontal Gene Transfer: Copy a block from P1, insert into P2
    elif roll < 0.97:  # 0.85 + 0.12 = 0.97
        # We need a source block (4-16 instructions)
        block_size = random.randint(4, 16)
        if len(p1) <= block_size:
            segment = list(p1)  # Take whole thing if small
        else:
            start = random.randint(0, len(p1) - block_size)
            segment = p1[start: start + block_size]

        # Insert into P2
        target = list(p2)
        insert_pt = random.randint(0, len(target))
        return target[:insert_pt] + segment + target[insert_pt:]

    # --- MECHANISM 3: GENE SHUFFLING (3%) ---
    # The Nuclear Option: Randomly pick instruction from A or B for each line
    else:
        child = []
        max_len = max(len(p1), len(p2))
        for i in range(max_len):
            # If both have a line here, flip a coin
            if i < len(p1) and i < len(p2):
                child.append(list(p1[i]) if random.random() < 0.5 else list(p2[i]))
            # If only P1 exists here
            elif i < len(p1):
                child.append(list(p1[i]))
            # If only P2 exists here
            else:
                child.append(list(p2[i]))
        return child


def get_grid_index(metrics):
    m_vals = [metrics.get("Activity", 0), metrics.get("Responsiveness", 0),
              metrics.get("ChannelEntropy", 0), metrics.get("AlgoDensity", 0)]
    idx = []
    for val, bins in zip(m_vals, GRID_BINS):
        b = min(int(val * bins), bins - 1)
        idx.append(b)
    return tuple(idx)


# ==============================================================================
# HARVEST & RUN
# ==============================================================================
def harvest_phase_1():
    print(f"🚜 Harvesting Phase 1 Archives (Target Batch: {SOURCE_BATCH})...")
    pool = []
    for i in range(10):
        path = f"World_{i}/backup_W{i}_{SOURCE_BATCH}.json.gz"
        if os.path.exists(path):
            try:
                with gzip.open(path, 'rt') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        pool.append(v["genome"])
                print(f"   --> World {i}: Loaded {len(data)} elites.")
            except:
                print(f"   --> World {i}: Error reading file.")
    return pool


def main():
    print("🚀 INITIATING PHASE 2: GENERATIONAL PULSE HYBRIDIZATION")

    # 1. SETUP
    run_dir = "World_Phase2"
    os.makedirs(run_dir, exist_ok=True)
    metrics_file = os.path.join(run_dir, "Metrics_Phase2.txt")

    # 2. HARVEST
    master_pool = harvest_phase_1()
    if not master_pool:
        print("❌ No organisms found. Aborting.")
        return
    print(f"🧬 Initial Gene Pool: {len(master_pool)} Organisms")

    # 3. GENERATIONAL LOOP
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

        for round_id in range(1, ROUNDS + 1):
            print(f"\n⚡ ROUND {round_id}/{ROUNDS} STARTING...")
            round_start = time.time()

            # --- A. BREEDING PULSE ---
            # Generate massive offspring pool (No evaluation yet)
            offspring = []
            print(f"   💕 Breeding {PAIRS_PER_ROUND} pairs -> {PAIRS_PER_ROUND * CHILDREN_PER_PAIR} children...")

            for _ in range(PAIRS_PER_ROUND):
                p1 = random.choice(master_pool)
                p2 = random.choice(master_pool)
                # Force 5 children per pair
                for _ in range(CHILDREN_PER_PAIR):
                    child = mutate(crossover(p1, p2))
                    offspring.append(child)

            # Combine Parents + Children for the Crucible
            # (Parents must re-prove themselves in the new physics)
            contestants = master_pool + offspring
            print(f"   🔥 The Crucible: Evaluating {len(contestants)} organisms...")

            # --- B. THE CRUCIBLE (MASSIVE PARALLEL EVALUATION) ---
            chunk_size = max(1, len(contestants) // NUM_WORKERS)
            chunks = [contestants[i:i + chunk_size] for i in range(0, len(contestants), chunk_size)]
            futures = [executor.submit(worker, c) for c in chunks]

            # --- C. THE FILTER (MAP-ELITES SELECTION) ---
            new_archive = {}
            total_processed = 0

            for f in concurrent.futures.as_completed(futures):
                results = f.result()
                total_processed += len(results)
                for genome, fit, mets in results:
                    if not mets: continue
                    idx = get_grid_index(mets)
                    if idx not in new_archive or fit > new_archive[idx]["fitness"]:
                        new_archive[idx] = {"genome": genome, "fitness": fit, "metrics": mets}

            # --- D. REPORTING ---
            elapsed = time.time() - round_start
            best_fit = max([v["fitness"] for v in new_archive.values()]) if new_archive else 0

            print(f"   ✅ Round Complete: {elapsed:.1f}s")
            print(f"   🏆 New Elite Archive: {len(new_archive)} survivors. Best Fit: {best_fit:.4f}")

            # Update Master Pool for next round
            # We ONLY keep the elites. The losers are discarded.
            master_pool = [v["genome"] for v in new_archive.values()]

            # Log
            with open(metrics_file, "a") as log:
                log.write(f"{round_id}, {elapsed:.1f}, {len(new_archive)}, {best_fit:.4f}\n")

            # Save Backup
            fname = os.path.join(run_dir, f"backup_Phase2_Round{round_id}.json.gz")
            with gzip.open(fname, 'wt', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in new_archive.items()}, f)

    print("🏁 PHASE 2 COMPLETE.")


if __name__ == "__main__":
    main()
