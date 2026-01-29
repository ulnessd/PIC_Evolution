import subprocess
import sys
import time

# ==============================================================================
# MULTIVERSE CONFIGURATION (The Gradient Strategy)
# ==============================================================================
worlds = [
    # --- GROUP A: THE SPRINTERS (Speed Pressure) ---
    # Goal: Evolve fast, lean logic.
    {"id": 0, "cycles": 2000, "noise": 0.02},
    {"id": 1, "cycles": 3000, "noise": 0.02},
    {"id": 2, "cycles": 4000, "noise": 0.02},

    # --- GROUP B: THE SCHOLARS (Memory Pressure) ---
    # Goal: Force delay loops / memory.
    {"id": 3, "cycles": 20000, "noise": 0.02},
    {"id": 4, "cycles": 24000, "noise": 0.02},
    {"id": 5, "cycles": 30000, "noise": 0.02},

    # --- GROUP C: THE LISTENERS (Noise Pressure) ---
    # Goal: Force input filtering / debouncing.
    {"id": 6, "cycles": 12000, "noise": 0.05},
    {"id": 7, "cycles": 12000, "noise": 0.08},
    {"id": 8, "cycles": 12000, "noise": 0.10},

    # --- GROUP D: THE CONTROL (Baseline) ---
    {"id": 9, "cycles": 12000, "noise": 0.02},
]

def main():
    processes = []
    print(f"🚀 LAUNCHING {len(worlds)} PARALLEL WORLDS...")
    print(f"--------------------------------------------------")

    for w in worlds:
        # Construct the command line arguments
        cmd = [
            sys.executable, "master_evolution.py",
            "--id", str(w["id"]),
            "--cycles", str(w["cycles"]),
            "--noise", str(w["noise"])
        ]
        
        # Spawn the process (Non-blocking)
        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"   [Started] World {w['id']} (PID: {p.pid}) | Cycles: {w['cycles']} | Noise: {w['noise']}")
        time.sleep(1) # Slight stagger to prevent IO spike

    print(f"--------------------------------------------------")
    print("✅ All worlds launched. Monitoring...")

    # Wait for all worlds to finish
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\n⚠️  KILL SIGNAL RECEIVED. TERMINATING ALL WORLDS...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()
