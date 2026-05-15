import os

class DataLogger:
    """
    Unified logger that writes ALL logs into a single file inside results/.

    Final log file name format:
        results/{model_name}_{schedule_mode}.log

    Example:
        results/gpt2-medium_spatial.log
    """

    def __init__(self, model_name="model", schedule_mode="spatial",
                 result_dir="results"):

        # Ensure directory exists
        os.makedirs(result_dir, exist_ok=True)

        # One log file per (model, schedule)
        base_name = f"{model_name}_{schedule_mode}"
        self.filename = os.path.join(result_dir, f"{base_name}.log")
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # Clear log at start of entire program run
        with open(self.filename, "w") as f:
            pass

    # -------------------------------------------------------------
    # Internal writer
    # -------------------------------------------------------------
    def _write(self, text):
        with open(self.filename, "a") as f:
            f.write(text + "\n")

    # -------------------------------------------------------------
    # Logging functions
    # -------------------------------------------------------------
    def log_separator(self, seq_len):
        self._write(f"\n========== New Run: seq_len = {seq_len} ==========\n")

    def log_node_cycles(self, node_name, start_cycle, end_cycle):
        self._write(f"Node: {node_name}, Start Cycle: {start_cycle}, End Cycle: {end_cycle}")

    def log_node_macs(self, node_name, macs):
        self._write(f"Node: {node_name}, MACs: {macs}")

    def log_fu_allocation(self, fu_name, m_value):
        self._write(f"FU: {fu_name}, M: {m_value}")

    def log_latency(self, latency):
        self._write(f"Final Latency: {latency}")

    def log_constraint(self, name, expr):
        self._write(f"Constraint: {name}, Expr: {expr}")

    def log_solver_status(self, status, termination):
        self._write(f"Solver Status: {status}, Termination: {termination}")

    def log_message(self, message):
        self._write(message)
