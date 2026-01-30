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
NUM_WORKERS = max(1, os.cpu_count() - 2)         # Devote almost entire CPU to the active world
BATCHES_PER_WORLD = 25_000  # How long to run each world before switching

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
    Evaluates a single genome by executing it on the frozen Pic10Sim substrate.

    IMPORTANT CONTRACT (Phase 1 / Part 1):
      * The cosmos bitstream is injected on GP3 (gp3_input).
      * GP0/GP1 are the two output bits (read as a 2-bit integer 0..3).
      * GP2 is reserved and held at 0 for now.
      * The simulator is driven ONLY via Pic10Sim.emulate_cycle(); no direct RAM writes
        and no manual stepping are permitted.

    Calculates ChannelEntropy from the realized output stream (compressibility proxy).
    """
    sim = Pic10Sim()
    sim.reset()

    # ----------------------------------------------------------------------
    # GENOME VALIDATION (STRICT)
    # ----------------------------------------------------------------------
    # We explicitly forbid malformed instructions. This avoids wasting compute
    # on garbage genomes and prevents any accidental over-emphasis of NOP.
    #
    # Policy: If the genome is malformed, return fitness=0 and empty metrics,
    # so the candidate cannot enter the archive.
    #
    # A valid genome is a list of 2-element sequences [op_id, operand] where:
    #   * op_id is an int in [0, len(OPCODE_LIST)-1]
    #   * operand is an int in [0,255]
    #   * genome length >= 1
    if not genome or not isinstance(genome, (list, tuple)):
        return 0.0, {}

    for instr in genome:
        if not isinstance(instr, (list, tuple)) or len(instr) != 2:
            return 0.0, {}
        try:
            op_id = int(instr[0])
            operand = int(instr[1])
        except Exception:
            return 0.0, {}
        if op_id < 0 or op_id >= len(OPCODE_LIST):
            return 0.0, {}
        if operand < 0 or operand > 255:
            return 0.0, {}

    # Load genome into the emulator once (program is a list of [opcode_id, operand] pairs)
    sim.load(genome, OPCODE_LIST)

    # Generate Physics
    cosmos = Cosmos(PHYSICS_CYCLES)

    score = 0
    valid_cycles = 0
    grace = 0
    last_target = 0

    toggles = 0
    last_out = 0
    outputs_history = []

    crashed = False

    for t in range(PHYSICS_CYCLES):
        bit = cosmos.inputs[t]
        target = cosmos.targets[t]

        # Drive the emulator for one instruction step
        out, did_crash = sim.emulate_cycle(gp2_input=0, gp3_input=bit)
        if did_crash:
            crashed = True
            break

        outputs_history.append(out)

        if out != last_out:
            toggles += 1
        last_out = out

        if target != last_target:
            grace = 16
        last_target = target

        if grace > 0:
            grace -= 1
        else:
            valid_cycles += 1
            if out == target:
                score += 1

    # Fitness (if crashed, treat as non-viable)
    if crashed:
        fitness = 0.0
    else:
        fitness = score / max(1, valid_cycles)

    # Channel entropy via compressibility of the output stream
    if outputs_history:
        try:
            as_bytes = bytes(outputs_history)
            compressed = zlib.compress(as_bytes)
            entropy = len(compressed) / len(outputs_history)
        except Exception:
            entropy = 0.0
    else:
        entropy = 0.0

    genome_len = len(genome)

    metrics = {
        "Activity": toggles / max(1, PHYSICS_CYCLES),
        "Responsiveness": fitness,
        "ChannelEntropy": entropy,
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
    # Rule 0: Safety Check
    if len(p1) < 2 or len(p2) < 2: return p1

    roll = random.random()

    # --- MECHANISM 1: HOMOLOGOUS CROSSOVER (85%) ---
    # Standard "Cut and Splice"
    if roll < 0.85:
        pt = random.randint(1, min(len(p1), len(p2)) - 1)
        # Deep-copy to avoid any accidental parent/child aliasing.
        raw_child = (p1[:pt] + p2[pt:])[:MAX_LINES]
        return [list(g) for g in raw_child]

    # --- MECHANISM 2: TRANSPOSON INSERTION (12%) ---
    # Copy a block from P1, insert into P2
    elif roll < 0.97:
        block_size = random.randint(4, 16)
        if len(p1) <= block_size:
            segment = [list(g) for g in p1]
        else:
            start = random.randint(0, len(p1) - block_size)
            segment = [list(g) for g in p1[start: start + block_size]]

        target = [list(g) for g in p2]
        insert_pt = random.randint(0, len(target))
        child = target[:insert_pt] + segment + target[insert_pt:]
        return child[:MAX_LINES]  # Cap size

    # --- MECHANISM 3: GENE SHUFFLING (3%) ---
    # The Nuclear Option: Randomly pick instruction from A or B for each line
    else:
        child = []
        max_len = max(len(p1), len(p2))
        for i in range(max_len):
            if i < len(p1) and i < len(p2):
                child.append(list(p1[i]) if random.random() < 0.5 else list(p2[i]))
            elif i < len(p1):
                child.append(list(p1[i]))
            else:
                child.append(list(p2[i]))
        return child[:MAX_LINES]


# ==============================================================================
# MAIN LOOP (SERIAL EXECUTION)
# ==============================================================================
def main():
    """
    PART 3 (Provenance + Diagnostics + Saving) ONLY

    This replacement modifies ONLY main():
      - Improved terminal output (elapsed time, crash frac, genome length stats, sexual markers)
      - Clean metrics CSV file format
      - Backup JSON.gz now wrapped with meta + archive
      - No silent save failures
    """
    import uuid
    import hashlib
    import platform
    from datetime import datetime

    global PHYSICS_CYCLES, NOISE_RATE  # Updated per-world

    # ---- Run identity (shared across all worlds) ----
    run_uuid = str(uuid.uuid4())
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    host = platform.node()

    # Hash the opcode list to lock provenance
    opcode_blob = ("\n".join(OPCODE_LIST)).encode("utf-8")
    opcode_hash = hashlib.sha256(opcode_blob).hexdigest()[:16]

    print(f"🚀 STARTING SERIAL EXECUTION ON {NUM_WORKERS} CORES")
    print(f"   Run UUID: {run_uuid}")
    print(f"   Timestamp: {run_timestamp}")
    print(f"   Host: {host}")
    print(f"   OpcodeHash: {opcode_hash}")
    print("")

    for config in WORLD_QUEUE:
        # ----------------------------
        # 1) SETUP WORLD
        # ----------------------------
        world_id = config["id"]
        PHYSICS_CYCLES = config["cycles"]
        NOISE_RATE = config["noise"]

        run_dir = f"World_{world_id}"
        os.makedirs(run_dir, exist_ok=True)

        metrics_file = os.path.join(run_dir, f"Metrics_W{world_id}.csv")
        backup_prefix = os.path.join(run_dir, f"backup_W{world_id}")

        print("\n========================================")
        print(f"🌍 SWITCHING TO WORLD {world_id}")
        print(f"   Cycles: {PHYSICS_CYCLES} | Noise: {NOISE_RATE}")
        print("========================================")

        # ----------------------------
        # 2) INITIALIZE ARCHIVE
        # ----------------------------
        archive = {}
        initial_pop = [[random_instruction() for _ in range(random.randint(5, 20))] for _ in range(1000)]

        # ----------------------------
        # 3) CREATE PROCESS POOL
        # ----------------------------
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

            # ---- SEEDING ----
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

            # ----------------------------
            # 4) METRICS FILE HEADER (new clean format)
            # ----------------------------
            if not os.path.exists(metrics_file):
                with open(metrics_file, "w") as f:
                    f.write(
                        "batch,elapsed_s,world_id,cycles,noise,"
                        "pop,best_fit,rate_per_s,mut_rate,"
                        "crash_frac,mean_len,min_len,max_len\n"
                    )

            # Base metadata for backups
            base_meta = {
                "run_uuid": run_uuid,
                "timestamp": run_timestamp,
                "host": host,
                "world_id": world_id,
                "world_physics": {"cycles": PHYSICS_CYCLES, "noise": NOISE_RATE},
                "input_contract": "GP3=cosmos_input, GP2=reserved, GP0/GP1=outputs",
                "opcode_hash": opcode_hash,
                "evolution_params": {
                    "BATCHES_PER_WORLD": BATCHES_PER_WORLD,
                    "GRID_SIZE": GRID_BINS,
                    "BATCH_PCT": BATCH_PCT,
                    "MIN_BATCH": MIN_BATCH,
                    "MAX_BATCH": MAX_BATCH,
                    "BASE_MUTATION_RATE": BASE_MUTATION_RATE,
                    "GAMMA_INTERVAL": GAMMA_INTERVAL,
                    "GAMMA_DURATION": GAMMA_DURATION,
                    "GAMMA_MUTATION_RATE": GAMMA_MUTATION_RATE,
                    "SEXUAL_EPOCH_INTERVAL": SEXUAL_EPOCH_INTERVAL,
                    "SEXUAL_POP_PCT": SEXUAL_POP_PCT,
                    "MAX_LINES": MAX_LINES,
                },
            }

            # ----------------------------
            # 5) EVOLUTION LOOP
            # ----------------------------
            start_time = time.time()
            total_orgs = 0

            for current_batch in range(1, BATCHES_PER_WORLD + 1):

                # Gamma logic
                cycle_pos = current_batch % GAMMA_INTERVAL
                is_gamma = cycle_pos < GAMMA_DURATION
                mut_rate = GAMMA_MUTATION_RATE if is_gamma else BASE_MUTATION_RATE

                archive_list = list(archive.values())
                if not archive_list:
                    print(f"   [W{world_id}] Archive empty; stopping world.")
                    break

                target_size = max(MIN_BATCH, min(MAX_BATCH, int(len(archive) * BATCH_PCT)))
                parents = []

                # ---- ASEXUAL ----
                for _ in range(target_size):
                    parents.append(mutate_genome(random.choice(archive_list)["genome"], mut_rate))

                # ---- SEXUAL (epochal mixing) ----
                did_sexual = False
                sexual_added = 0
                if current_batch % SEXUAL_EPOCH_INTERVAL == 0:
                    did_sexual = True
                    sexual_added = int(len(archive) * SEXUAL_POP_PCT)
                    for _ in range(sexual_added):
                        p1 = random.choice(archive_list)["genome"]
                        p2 = random.choice(archive_list)["genome"]
                        parents.append(mutate_genome(crossover_homologous(p1, p2), mut_rate))

                # ---- EXECUTION ----
                total_orgs += len(parents)
                chunk_size = max(1, len(parents) // NUM_WORKERS)
                chunks = [parents[i:i + chunk_size] for i in range(0, len(parents), chunk_size)]
                futures = [executor.submit(worker, c) for c in chunks]

                # Track crashes / invalids in this batch (mets == {} means invalid/crash)
                batch_total = 0
                batch_valid = 0

                for f in concurrent.futures.as_completed(futures):
                    for res in f.result():
                        batch_total += 1
                        prog, fit, mets = res
                        if mets:
                            batch_valid += 1
                            idx = get_grid_index(mets)
                            if idx not in archive or fit > archive[idx]["fitness"]:
                                archive[idx] = {"genome": prog, "fitness": fit, "metrics": mets}

                crash_frac = 1.0 - (batch_valid / batch_total) if batch_total > 0 else 0.0

                # ---- Compute length stats over archive ----
                if archive:
                    lengths = [len(v["genome"]) for v in archive.values()]
                    mean_len = sum(lengths) / len(lengths)
                    min_len = min(lengths)
                    max_len = max(lengths)
                    best_fit = max(v["fitness"] for v in archive.values())
                else:
                    mean_len, min_len, max_len, best_fit = 0.0, 0, 0, 0.0

                elapsed = time.time() - start_time
                rate = total_orgs / elapsed if elapsed > 0 else 0.0

                # ---- Terminal reporting ----
                if current_batch % 10 == 0 or did_sexual or current_batch == 1:
                    if did_sexual:
                        print(f"   [W{world_id}] [SEXUAL EPOCH] added {sexual_added} recombinants")

                    print(
                        f"   [W{world_id}] Batch {current_batch:5d} | "
                        f"Elapsed {elapsed:7.1f}s | Pop={len(archive):4d} | "
                        f"BestFit={best_fit:.4f} | Rate={rate:8.0f}/s | Mut={mut_rate:.3f} | "
                        f"CrashFrac={crash_frac:.3f} | Len(mean/min/max)={mean_len:.2f}/{min_len}/{max_len}"
                    )

                # ---- Metrics CSV ----
                with open(metrics_file, "a") as log:
                    log.write(
                        f"{current_batch},{elapsed:.3f},{world_id},{PHYSICS_CYCLES},{NOISE_RATE},"
                        f"{len(archive)},{best_fit:.6f},{rate:.3f},{mut_rate:.6f},"
                        f"{crash_frac:.6f},{mean_len:.6f},{min_len},{max_len}\n"
                    )

                # ---- Backup ----
                if current_batch % BACKUP_INTERVAL_BATCHES == 0:
                    fname = f"{backup_prefix}_{current_batch}.json.gz"

                    # Serialize archive keys as strings (stable + human-inspectable)
                    archive_payload = {str(k): v for k, v in archive.items()}

                    meta = dict(base_meta)
                    meta.update({
                        "batch_completed": current_batch,
                        "elapsed_s": elapsed,
                        "archive_pop": len(archive),
                        "best_fit": best_fit,
                        "mut_rate": mut_rate,
                    })

                    payload = {"meta": meta, "archive": archive_payload}

                    with gzip.open(fname, "wt", encoding="utf-8") as f:
                        json.dump(payload, f)

                    print(f"   [W{world_id}] SAVED {os.path.basename(fname)} (pop={len(archive)}, best={best_fit:.4f})")

        print(f"✅ WORLD {world_id} COMPLETE.")



if __name__ == "__main__":
    main()
