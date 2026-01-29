import random
import concurrent.futures
import os
import json
import gzip
import zlib
import time
import copy
import sys
import math

# ==============================================================================
# CONFIGURATION: THE SERIAL STEAMROLLER
# ==============================================================================
# HARDWARE SETTINGS
NUM_WORKERS = 30          # Devote almost entire CPU to the active world
BATCHES_PER_WORLD = 200  # How long to run each world before switching

# THE QUEUE (The script will run these in order)
WORLD_QUEUE = [
    # A (Sprinters)
    {"id": 0, "cycles": 2000, "noise": 0.02},
    {"id": 1, "cycles": 3000, "noise": 0.02},
    {"id": 2, "cycles": 4000, "noise": 0.02},
    # B (Scholars)
    {"id": 3, "cycles": 20000, "noise": 0.02},
    {"id": 4, "cycles": 24000, "noise": 0.02},
    {"id": 5, "cycles": 30000, "noise": 0.02},
    # C (Listeners)
    {"id": 6, "cycles": 12000, "noise": 0.05},
    {"id": 7, "cycles": 12000, "noise": 0.08},
    {"id": 8, "cycles": 12000, "noise": 0.10},
    # D (Control)
    {"id": 9, "cycles": 12000, "noise": 0.02},
]

# GLOBAL VARIABLES (Updated dynamically by Main)
PHYSICS_CYCLES = 12000
NOISE_RATE = 0.02

# CONSTANTS
GAP_MIN, GAP_MAX = 4, 20
COHERENCE_TIME = 150
GAMMA_INTERVAL = 25000
GAMMA_DURATION = 200
GAMMA_MUTATION_RATE = 0.30
BASE_MUTATION_RATE = 0.05

# SYSTEM
GRID_BINS = [16, 16, 16, 16]
MIN_BATCH = 500
MAX_BATCH = 6000
BATCH_PCT = 0.10
SEXUAL_POP_PCT = 0.50
SEXUAL_EPOCH_INTERVAL = 20
MAX_LINES = 256
BACKUP_INTERVAL_BATCHES = 100

# PIC10SIM IMPORT
try:
    from Pic10Sim import Pic10Sim
except ImportError:
    print("CRITICAL ERROR: Pic10Sim.py not found.")
    sys.exit()

# OPCODE LIST
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
# THE COSMOS: PHYSICS ENGINE
# ==============================================================================
class Cosmos:
    """
    Generates the 12,000 cycle test pattern with Physics.
    """

    def __init__(self, length):
        self.length = length
        self.inputs = []
        self.targets = []
        self._generate()

    def _generate(self):
        # Targets: 0=Silence, 1=Hum, 2=Duet, 3=Warble
        # Patterns (8-bit)
        PATTERNS = {
            1: [1, 1, 1, 1, 0, 0, 0, 0],  # Hum
            2: [1, 1, 0, 0, 1, 1, 0, 0],  # Duet
            3: [1, 0, 1, 0, 1, 0, 1, 0],  # Warble
            0: [0, 0, 0, 0, 0, 0, 0, 0]  # Silence
        }

        t = 0
        rts_state = 0  # Phase State for Noise

        while t < self.length:
            # 1. Choose Signal
            target = random.choice([0, 1, 2, 3])
            pattern = PATTERNS[target]

            # 2. Polarity Inversion (The Mirror)
            invert = (random.random() < 0.5)

            # 3. Burst Duration
            # Approx 150 cycles, aligned to 8-bit packets
            num_packets = random.randint(15, 25)

            for _ in range(num_packets):
                for bit in pattern:
                    if t >= self.length: break

                    # Apply Polarity
                    val = (1 - bit) if invert else bit

                    # Apply Phase Noise (The Fog)
                    if random.random() < NOISE_RATE:
                        rts_state = 1 - rts_state  # Flip phase state

                    # Output is XORed with noise state
                    val = val ^ rts_state

                    self.inputs.append(val)
                    self.targets.append(target)
                    t += 1

            # 4. The Gap (Variable Silence)
            gap = random.randint(GAP_MIN, GAP_MAX)
            for _ in range(gap):
                if t >= self.length: break
                self.inputs.append(0)
                self.targets.append(0)
                t += 1


# ==============================================================================
# DRIVER ENGINE (The Interface)
# ==============================================================================
def evaluate_organism(genome):
    """
    Manually drives the Pic10Sim to avoid internal loop bugs.
    Calculates REAL Entropy to fix the Population Stagnation.
    """
    sim = Pic10Sim()
    sim.reset()

    # Generate Physics
    cosmos = Cosmos(PHYSICS_CYCLES)

    # Run Loop
    score = 0
    valid_cycles = 0
    grace = 0
    last_target = 0

    toggles = 0
    last_out = 0

    # DATA STREAM FOR ENTROPY (The Fix)
    outputs_history = []

    genome_len = len(genome)
    if genome_len == 0: return 0.0, {}

    # Pre-calculate string ops to save time
    decoded_prog = []
    for instr in genome:
        op_id, raw_operand = instr
        if op_id >= len(OPCODE_LIST): op_id = 22  # NOP fallback
        mnemonic = OPCODE_LIST[op_id]

        bit_index = 0
        operand = raw_operand
        if mnemonic in ['BCF', 'BSF', 'BTFSC', 'BTFSS']:
            bit_index = (raw_operand >> 5) & 0x07
            operand = raw_operand & 0x1F

        decoded_prog.append((mnemonic, operand, bit_index))

    # Execute
    for t in range(PHYSICS_CYCLES):
        inp = cosmos.inputs[t]
        target = cosmos.targets[t]

        # --- 1. INJECT INPUT (GP2) ---
        current_gpio = sim.ram[0x06]
        if inp:
            sim.ram[0x06] = (current_gpio & ~0x04) | 0x04  # Set Bit 2
        else:
            sim.ram[0x06] = (current_gpio & ~0x04)  # Clear Bit 2

        # --- 2. FETCH & STEP ---
        pc = sim.pc
        if pc >= genome_len: pc %= genome_len

        mnemonic, op, bit = decoded_prog[pc]
        sim.step(mnemonic, op, bit)

        # --- 3. READ OUTPUT (GP0, GP1) ---
        out = sim.ram[0x06] & 0x03
        outputs_history.append(out)  # Capture for Entropy

        # Metrics
        if out != last_out: toggles += 1
        last_out = out

        # Scoring
        if target != last_target:
            grace = 16
        last_target = target

        if grace > 0:
            grace -= 1
        else:
            valid_cycles += 1
            if out == target: score += 1

    # Finalize
    fitness = score / max(1, valid_cycles)

    # --- ENTROPY CALCULATION (The Fix) ---
    if outputs_history:
        # Compress the output stream to measure complexity
        compressed = zlib.compress(bytes(outputs_history))
        entropy = len(compressed) / len(outputs_history)
    else:
        entropy = 0.0

    metrics = {
        "Activity": toggles / PHYSICS_CYCLES,
        "Responsiveness": fitness,
        "ChannelEntropy": entropy,  # Now uses Real Data
        "AlgoDensity": genome_len / MAX_LINES
    }

    return fitness, metrics


def get_grid_index(metrics):
    def map_dim(val, steps):
        return min(int(val * steps), steps - 1)

    return (
        map_dim(metrics.get("Activity", 0), GRID_BINS[0]),
        map_dim(metrics.get("Responsiveness", 0), GRID_BINS[1]),
        map_dim(metrics.get("ChannelEntropy", 0), GRID_BINS[2]),
        map_dim(metrics.get("AlgoDensity", 0), GRID_BINS[3])
    )


def worker(population_chunk):
    results = []
    for genome in population_chunk:
        try:
            fit, mets = evaluate_organism(genome)
            results.append((genome, fit, mets))
        except Exception:
            results.append((genome, 0.0, {}))
    return results


# ==============================================================================
# GENETIC OPERATORS
# ==============================================================================
def random_instruction():
    # Return [OpcodeID, Operand]
    return [random.randint(0, len(OPCODE_LIST) - 1), random.randint(0, 255)]


def mutate_genome(genome, rate):
    new_genome = copy.deepcopy(genome)
    # Point Mutation
    for i in range(len(new_genome)):
        if random.random() < rate:
            if random.random() < 0.5:
                new_genome[i][0] = random.randint(0, len(OPCODE_LIST) - 1)
            else:
                new_genome[i][1] = random.randint(0, 255)

    # Indel
    if random.random() < (rate * 0.5):
        if len(new_genome) < MAX_LINES and random.random() < 0.5:
            new_genome.insert(random.randint(0, len(new_genome)), random_instruction())
        elif len(new_genome) > 1:
            new_genome.pop(random.randint(0, len(new_genome) - 1))

    return new_genome


def crossover_homologous(p1, p2):
    if len(p1) < 2 or len(p2) < 2: return p1
    pt = random.randint(1, min(len(p1), len(p2)) - 1)
    return (p1[:pt] + p2[pt:])[:MAX_LINES]


# ==============================================================================
# MAIN LOOP (SERIAL EXECUTION)
# ==============================================================================
def main():
    global PHYSICS_CYCLES, NOISE_RATE  # We update these for each world

    print(f"🚀 STARTING SERIAL EXECUTION ON {NUM_WORKERS} CORES")

    for config in WORLD_QUEUE:
        # 1. SETUP WORLD
        world_id = config["id"]
        PHYSICS_CYCLES = config["cycles"]
        NOISE_RATE = config["noise"]

        run_dir = f"World_{world_id}"
        os.makedirs(run_dir, exist_ok=True)
        metrics_file = os.path.join(run_dir, f"Metrics_W{world_id}.txt")
        backup_prefix = os.path.join(run_dir, f"backup_W{world_id}")

        print(f"\n========================================")
        print(f"🌍 SWITCHING TO WORLD {world_id}")
        print(f"   Cycles: {PHYSICS_CYCLES} | Noise: {NOISE_RATE}")
        print(f"========================================")

        # 2. INITIALIZE ARCHIVE
        archive = {}
        initial_pop = [[random_instruction() for _ in range(random.randint(5, 20))] for _ in range(1000)]

        # 3. CREATE PROCESS POOL (Fresh pool for each world to ensure globals update)
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

            # SEEDING
            print("   Seeding population...")
            chunk_size = max(1, len(initial_pop) // NUM_WORKERS)
            chunks = [initial_pop[i:i + chunk_size] for i in range(0, len(initial_pop), chunk_size)]
            futures = [executor.submit(worker, c) for c in chunks]

            for f in concurrent.futures.as_completed(futures):
                for res in f.result():
                    prog, fit, mets = res
                    if mets:
                        idx = get_grid_index(mets)
                        if idx not in archive or fit > archive[idx]["fitness"]:
                            archive[idx] = {"genome": prog, "fitness": fit, "metrics": mets}

            # LOGGING HEADER
            if not os.path.exists(metrics_file):
                with open(metrics_file, "w") as f:
                    f.write("Batch, Time, Pop, BestFit, Rate, MutRate\n")

            # 4. EVOLUTION LOOP
            start_time = time.time()
            total_orgs = 0

            for current_batch in range(1, BATCHES_PER_WORLD + 1):
                # Gamma Logic
                cycle_pos = current_batch % GAMMA_INTERVAL
                is_gamma = cycle_pos < GAMMA_DURATION
                mut_rate = GAMMA_MUTATION_RATE if is_gamma else BASE_MUTATION_RATE

                # Reproduction
                archive_list = list(archive.values())
                if not archive_list: break

                target_size = max(MIN_BATCH, min(MAX_BATCH, int(len(archive) * BATCH_PCT)))
                parents = []

                # Asexual
                for _ in range(target_size):
                    parents.append(mutate_genome(random.choice(archive_list)["genome"], mut_rate))

                # Sexual
                if current_batch % SEXUAL_EPOCH_INTERVAL == 0:
                    for _ in range(int(len(archive) * SEXUAL_POP_PCT)):
                        p1 = random.choice(archive_list)["genome"]
                        p2 = random.choice(archive_list)["genome"]
                        parents.append(mutate_genome(crossover_homologous(p1, p2), mut_rate))

                # Execution
                total_orgs += len(parents)
                chunk_size = max(1, len(parents) // NUM_WORKERS)
                chunks = [parents[i:i + chunk_size] for i in range(0, len(parents), chunk_size)]
                futures = [executor.submit(worker, c) for c in chunks]

                for f in concurrent.futures.as_completed(futures):
                    for res in f.result():
                        prog, fit, mets = res
                        if mets:
                            idx = get_grid_index(mets)
                            if idx not in archive or fit > archive[idx]["fitness"]:
                                archive[idx] = {"genome": prog, "fitness": fit, "metrics": mets}

                # Metrics & Reporting
                elapsed = time.time() - start_time
                best_fit = max([v["fitness"] for v in archive.values()]) if archive else 0
                rate = total_orgs / elapsed if elapsed > 0 else 0

                if current_batch % 10 == 0:
                    print(
                        f"   [W{world_id}] Batch {current_batch}/{BATCHES_PER_WORLD}: BestFit={best_fit:.4f}, Rate={rate:.0f}/s")
                    with open(metrics_file, "a") as log:
                        log.write(
                            f"{current_batch}, {elapsed:.1f}, {len(archive)}, {best_fit:.4f}, {rate:.1f}, {mut_rate}\n")

                if current_batch % BACKUP_INTERVAL_BATCHES == 0:
                    fname = f"{backup_prefix}_{current_batch}.json.gz"
                    try:
                        with gzip.open(fname, 'wt', encoding='utf-8') as f:
                            json.dump({str(k): v for k, v in archive.items()}, f)
                    except:
                        pass

        print(f"✅ WORLD {world_id} COMPLETE.")


if __name__ == "__main__":
    main()
