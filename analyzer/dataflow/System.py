import json
import matplotlib.pyplot as plt
import os

ARCH_DIR = "Scheduler"
HARD_DIR = "Hardware"
CONSTRAINT_DIR = "Constraints"
RESULT_DIR = "results"

from LLMconfig import TransformerModelConfig
from hardware import HardwareDevice
from LLMmodel import LayerGraph
from FU_mapping import HardwareModel
from datalogger import DataLogger
from constraint import Constraints
from Solver import ConstraintSolver


# ---------------------- Plot Functions ----------------------
def plot_fu_allocations(fu_allocations):
    filtered_allocations = {k: v for k, v in fu_allocations.items() if v > 1}
    if not filtered_allocations:
        print("No significant FU allocations to plot.")
        return

    sorted_allocations = dict(sorted(filtered_allocations.items()))
    names = list(sorted_allocations.keys())
    values = list(sorted_allocations.values())
    
    plt.figure(figsize=(15, 8))
    plt.barh(names, values, color='skyblue')
    plt.xlabel("Allocated Compute Power (M, MACs/cycle)")
    plt.ylabel("Functional Unit")
    plt.title("Optimal M Allocation (Seq Len 1024)")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_performance_vs_sequence(sequence_lengths, latencies, throughputs):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.plot(sequence_lengths, latencies, marker="o", label="Latency")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Throughput (tokens/s)")
    ax2.plot(sequence_lengths, throughputs, marker="x", linestyle="--", color="red")

    plt.title("Performance vs Sequence Length")
    fig.tight_layout()
    plt.show()


# =============================== SYSTEM ===============================
if __name__ == "__main__":

    # -----------------------------
    # Load model + hardware config
    # -----------------------------
    #model_name = "openai-community/gpt2-medium"
    model_name = "openai/gpt-oss-20b"
    schedule_mode = "spatial"

    model_config = TransformerModelConfig.from_hub(model_name)

    u280_device = HardwareDevice.from_json(
        file_path=os.path.join(HARD_DIR, "hardware_config.json"),
        device_name="Alveo_U280"
    )

    #arch_json_path = os.path.join(ARCH_DIR, "gpt2_arch_spatial.json")
    arch_json_path = os.path.join(ARCH_DIR, "gpt-oss_arch_spatial.json")
    arch_data = json.load(open(arch_json_path))

    # -----------------------------
    # Unified logger (ONE FILE ONLY)
    # -----------------------------
    logger = DataLogger(
        model_name=model_name,
        schedule_mode=schedule_mode,
        result_dir=RESULT_DIR
    )

    # -----------------------------
    # Sequence lengths to evaluate
    # -----------------------------
    sequence_lengths = [32, 64, 128, 256, 512, 1024, 2048]

    # ================================================================
    # STEP 1: Build IR blocks and produce constraints_all.json
    # ================================================================
    master_ir_dict = {}

    for seq_len in sequence_lengths:

        system_config = {
            "seq_len": seq_len,
            "layers_per_fpga": 1,
            "fifo_depth": 16,
            "pp_size": 1,
            "tp_size": 1,
            "bw_weight": 8,
            "bw_activation": 8,
            "reuse_factor": 1           # Default reuse factor
        }

        layer_graph = LayerGraph(arch_data, model_config, system_config)
        #print("[DEBUG] resolved_dims =", layer_graph.resolved_dims)
        reuse_factor = system_config.get('reuse_factor', 1)
        hardware_model = HardwareModel(layer_graph, reuse_factor=reuse_factor)

        constraints_obj = Constraints(
            u280_device, model_config, system_config, hardware_model, logger=logger
        )

        ir_block = constraints_obj.build_ir_block(seq_len)
        master_ir_dict[str(seq_len)] = ir_block

    Constraints.save_master_ir(
        filepath=os.path.join(CONSTRAINT_DIR, "constraints_all.json"),
        master_ir_dict=master_ir_dict
    )

    # ================================================================
    # STEP 2: Pyomo solver evaluation
    # ================================================================
    all_solutions = {}
    latencies_ms = []
    throughputs = []

    print("=" * 90)
    print(f"{'Seq Len':<10} | {'Latency (ms)':<15} | {'Throughput (tok/s)':<22} | {'M_total used':<15}")
    print("-" * 90)

    for seq_len in sequence_lengths:

        system_config = {
            "seq_len": seq_len,
            "layers_per_fpga": 1,
            "fifo_depth": 16,
            "pp_size": 1,
            "tp_size": 1,
            "bw_weight": 8,
            "bw_activation": 8,
        }

        layer_graph = LayerGraph(arch_data, model_config, system_config)
        hardware_model = HardwareModel(layer_graph)

        solver = ConstraintSolver(
            hardware_model=hardware_model,
            hardware_device=u280_device,
            constraint_path=os.path.join(CONSTRAINT_DIR, "constraints_all.json"),
            seq_len=seq_len,
            logger=logger
        )

        result = solver.run()

        Mi = result["solution"]
        latency_sec = result["latency_seconds"]

        all_solutions[seq_len] = result

        M_used = sum(Mi.values())
        tok_per_sec = seq_len / latency_sec if latency_sec > 0 else 0

        print(f"{seq_len:<10} | "
              f"{(latency_sec*1000):<15.3f} | "
              f"{tok_per_sec:<22.2f} | "
              f"{M_used:<15.1f}")

        latencies_ms.append(latency_sec * 1000)
        throughputs.append(tok_per_sec)

    print("=" * 90)
    print("--- Analysis Complete ---")
