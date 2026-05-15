import json
from hardware import HardwareDevice
from LLMconfig import TransformerModelConfig
from FU_mapping import HardwareModel
from memory_calculation import get_memory_footprint_calculator


# -------------------------------------------------------------------
# Constraint Expression Object
# -------------------------------------------------------------------
class ConstraintExpr:
    def __init__(self, name: str, expr: str):
        self.name = name
        self.expr = expr


# -------------------------------------------------------------------
# Unified Constraints Class
# -------------------------------------------------------------------
class Constraints:

    def __init__(self,
                 hardware_device: HardwareDevice,
                 model_config: TransformerModelConfig,
                 system_config: dict,
                 hardware_model: HardwareModel,
                 logger=None):

        self.hardware = hardware_device
        self.model_config = model_config
        self.system_config = system_config
        self.hardware_model = hardware_model
        self.num_layers = self.model_config.num_layers

        self.logger = logger

        # Runtime memory calculator (original functionality)
        self.memory_calculator = get_memory_footprint_calculator(
            self.hardware_model, self.model_config, self.system_config
        )

        # IR containers
        self.ir_constraints = []
        self.ir_parameters = {}

        # Build IR once during initialization
        self._build_ir_constraints()

    # ============================================================
    # =============== Runtime Post-Check Functions ================
    # ============================================================
    def check_total_macs(self, fus_with_m: dict) -> bool:
        """Check that the sum of allocated MACs does not exceed hardware limit."""
        total_m = sum(fu["M"] for fu in fus_with_m.values())
        return total_m <= self.hardware.total_macs

    def check_bram(self, fus_with_m: dict, architecture_type: str) -> bool:
        """
        Check BRAM usage. 
        Note: currently only meaningful for 'dataflow' architecture.
        """
        if architecture_type != "dataflow":
            return True

        C = self.system_config.get("layers_per_fpga", 1)
        bw_weight = self.system_config.get("bw_weight", 8)
        bw_bram = self.hardware.bitwidth_bram

        bram_per_m = bw_weight / bw_bram if bw_bram > 0 else float("inf")

        matmul_fus_r = 0
        other_fus_r = 0

        for fu_name, fu_details in fus_with_m.items():
            r_factor = self.hardware_model.fus.get(fu_name, {}).get('reuse_factor', 1)
            effective_m = fu_details["M"] / r_factor
            r_i = effective_m * bram_per_m

            if "matmul" in fu_details["type"]:
                matmul_fus_r += r_i
            else:
                other_fus_r += r_i


        required_bram = (other_fus_r + 2 * matmul_fus_r) * C
        return required_bram <= self.hardware.num_bram_blocks

    def check_memory_capacity(self, m_allocations: dict) -> bool:
        """Check that the memory footprint fits into on-chip memory."""
        memory_footprint = self.memory_calculator(m_allocations)
        total_memory_needed = sum(memory_footprint.values())
        return total_memory_needed <= self.hardware.on_chip_mem_bytes

    def check_memory_bandwidth(self, m_allocations: dict) -> bool:
        """Check off-chip memory bandwidth constraints."""
        bw_w_bytes = self.system_config.get("bw_weight", 8) / 8
        frequency_hz = self.hardware.frequency_hz

        total_m = sum(m_allocations.values())
        required_bw = total_m * bw_w_bytes * frequency_hz

        return required_bw <= self.hardware.off_chip_mem_bw_bytes_per_sec

    def check_all(self, m_allocations: dict, architecture_type: str = "dataflow"):
        """
        Run all runtime constraint checks after M allocation.
        This does NOT affect solver. It is only used for debugging.

        Now integrates your teammate's usage reporting (MAC/BRAM/Mem/BW).
        """

        #print("\n--- Verifying Constraints (Runtime Check with Usage) ---")

        # ===== Structured fus_with_m =====
        fus_with_m = {
            fu_name: {
                "type": self.hardware_model.fus.get(fu_name, {}).get("type", "unknown"),
                "M": m_value
            }
            for fu_name, m_value in m_allocations.items()
        }

        # =====================================================
        # 1. MAC Usage (added from teammate)
        # =====================================================
        total_m = sum(fu["M"] for fu in fus_with_m.values())
        print(f"[MAC Usage] {total_m:,.0f} / {self.hardware.total_macs:,.0f}  "
              f"({total_m / self.hardware.total_macs * 100:.2f}%)")
        if total_m > self.hardware.total_macs:
            print("  WARNING: Total MACs constraint NOT satisfied.")
        else:
            print("  OK: Total MACs constraint satisfied.")

        # =====================================================
        # 2. BRAM Usage (same logic, but added usage print)
        # =====================================================
        if architecture_type == "dataflow":
            C = self.system_config.get("layers_per_fpga", 1)
            bw_weight = self.system_config.get("bw_weight", 8)
            bw_bram = self.hardware.bitwidth_bram
            bram_per_m = bw_weight / bw_bram if bw_bram > 0 else float("inf")

            matmul_fus_r = 0
            other_fus_r = 0
            for fu_name, fu_details in fus_with_m.items():
                r_factor = self.hardware_model.fus.get(fu_name, {}).get('reuse_factor', 1)
                effective_m = fu_details["M"] / r_factor
                r_i = effective_m * bram_per_m

            total_bram_needed = (other_fus_r + 2 * matmul_fus_r) * C

            print(f"[BRAM Port Usage] {total_bram_needed:,.2f} / {self.hardware.num_bram_blocks:,.0f}  "
                  f"({total_bram_needed / self.hardware.num_bram_blocks * 100:.2f}%)")

            if total_bram_needed > self.hardware.num_bram_blocks:
                print("  WARNING: BRAM constraint NOT satisfied.")
            else:
                print("  OK: BRAM constraint satisfied.")
        else:
            print("[BRAM] Skip (not dataflow architecture)")

        # =====================================================
        # 3. On-chip Memory Capacity
        # =====================================================
        memory_footprint = self.memory_calculator(m_allocations)
        total_memory_needed = sum(memory_footprint.values())

        print(f"[On-chip Memory Usage] {total_memory_needed:,.0f} / {self.hardware.on_chip_mem_bytes:,.0f} bytes  "
              f"({total_memory_needed / self.hardware.on_chip_mem_bytes * 100:.2f}%)")

        if total_memory_needed > self.hardware.on_chip_mem_bytes:
            print("  WARNING: On-chip memory capacity constraint NOT satisfied.")
        else:
            print("  OK: On-chip memory capacity satisfied.")

        # =====================================================
        # 4. Off-chip Memory Bandwidth (with fallback selection)
        # =====================================================
        bw_w_bytes = self.system_config.get("bw_weight", 8) / 8
        frequency_hz = self.hardware.frequency_hz

        # Determine which FUs load weights (your fallback)
        weight_loading_fus = set()
        for node in self.hardware_model.layer_graph.nodes:
            flag = node.get("require_offchip_memory")
            if flag is None:
                mac = self.hardware_model.layer_graph.node_macs.get(node["name"], 0)
                flag = (mac > 0)

            if flag:
                fu_name = self.hardware_model.fu_map.get(node["name"])
                if fu_name:
                    weight_loading_fus.add(fu_name)

        total_m_for_weights = 0
        for fu in weight_loading_fus:
            M_val = m_allocations.get(fu, 0)
            r_factor = self.hardware_model.fus.get(fu, {}).get('reuse_factor', 1)
            total_m_for_weights += M_val / r_factor

        total_bw_needed = total_m_for_weights * bw_w_bytes * frequency_hz
        print(f"[Off-chip BW Usage] {total_bw_needed:,.0f} / "
              f"{self.hardware.off_chip_mem_bw_bytes_per_sec:,.0f} bytes/s  "
              f"({total_bw_needed / self.hardware.off_chip_mem_bw_bytes_per_sec * 100:.2f}%)")

        if total_bw_needed > self.hardware.off_chip_mem_bw_bytes_per_sec:
            print("  WARNING: Off-chip memory bandwidth NOT satisfied.")
        else:
            print("  OK: Off-chip memory bandwidth satisfied.")

        print("--- Verification Complete ---\n")


    # ============================================================
    # =============== IR Constraint Generation ====================
    # ============================================================
    def _build_ir_constraints(self):
        """Build solver IR constraints."""
        fu_list = list(self.hardware_model.fus.keys())

        self.ir_parameters["num_layers"] = self.num_layers
        self.ir_parameters["layers_per_fpga"] = self.system_config.get("layers_per_fpga", 1)

        # -------------------------------
        # Constraint 1: Total MAC budget
        # -------------------------------
        expr_mac = " + ".join(f"{fu}" for fu in fu_list)
        expr_mac += f" <= {self.hardware.total_macs}"
        self.ir_constraints.append(ConstraintExpr("total_macs_budget", expr_mac))

        # -------------------------------
        # Constraint 2: Off-chip bandwidth (linearized)
        # ----------------------------------------
        bw_w_bytes = self.system_config.get("bw_weight", 8) / 8
        freq = self.hardware.frequency_hz
        coeff = bw_w_bytes * freq

        weight_loading_fus = set()

        for node in self.hardware_model.layer_graph.nodes:

            # see if node requires off-chip memory  
            flag = node.get("require_offchip_memory")

            # ===== Fallback：if flag is None, add this node =====
            # if flag is None:
            #     mac = self.hardware_model.layer_graph.node_macs.get(node["name"], 0)
            #     flag = (mac > 0)

            # if requires off-chip memory, add its FU
            if flag:
                fu_name = self.hardware_model.fu_map.get(node["name"])
                if fu_name:
                    weight_loading_fus.add(fu_name)

        # # ===== DEBUG  =====
        # print("\n[DEBUG] --- BW Constraint Debug (Fallback Mode) ---")
        # for node in self.hardware_model.layer_graph.nodes:
        #     print(f"node={node['name']} flag={node.get('require_offchip_memory')} mac={self.hardware_model.layer_graph.node_macs.get(node['name'], 0)}")
        # print("[DEBUG] Selected FUs needing BW =", weight_loading_fus)
        # print("---------------------------------------------------\n")

        # ===== add BW Constraint in json file =====
        if weight_loading_fus:
            terms = []
            for fu in weight_loading_fus:
                r_factor = self.hardware_model.fus.get(fu, {}).get('reuse_factor', 1)
                coeff_fu = coeff / r_factor            # 注意这里 / r_factor
                terms.append(f"{coeff_fu} * {fu}")

            expr_bw = " + ".join(terms) + f" <= {self.hardware.off_chip_mem_bw_bytes_per_sec}"
            self.ir_constraints.append(ConstraintExpr("offchip_bandwidth", expr_bw))
            print("[DEBUG] Added BW constraint:", expr_bw)
        else:
            print("[WARNING] No FU marked as BW-loading — this should not happen now!")

        # -------------------------------
        # Constraint 3: BRAM capacity (approx linear model)
        # -------------------------------
        C = self.system_config.get("layers_per_fpga", 1)
        bw_weight = self.system_config.get("bw_weight", 8)
        bw_bram = self.hardware.bitwidth_bram
        bram_per_m = bw_weight / bw_bram if bw_bram > 0 else 1e9  # fallback

        terms = []
        for fu in fu_list:
            fu_type = self.hardware_model.fus[fu]["type"]
            r_factor = self.hardware_model.fus.get(fu, {}).get('reuse_factor', 1)
            scale = 2 if "matmul" in fu_type else 1
            coeff = scale * bram_per_m / r_factor
            terms.append(f"{coeff} * {fu}")

        # (other_r + 2 * matmul_r) * C <= num_bram
        #  =>  sum(coeff * M_fu) <= num_bram / C
        rhs = self.hardware.num_bram_blocks / max(C, 1)
        expr_bram = " + ".join(terms) + f" <= {rhs}"
        self.ir_constraints.append(ConstraintExpr("bram_capacity", expr_bram))
    
        # -------------------------------
        # Constraint 4: On-chip Memory Capacity (linear form)
        # -------------------------------

        # # Precompute M-independent memory
        # bw_w_bytes = self.system_config.get("bw_weight", 8) / 8
        # C = self.system_config.get("layers_per_fpga", 1)

        # # Compute M-independent memory (kv_cache + fifo_buffers)
        # baseline_M = {fu: 0.0 for fu in fu_list}
        # baseline_memory = self.memory_calculator(baseline_M)
        # const_mem = baseline_memory["kv_cache"] + baseline_memory["fifo_buffer"]

        # # Compute RHS available for weight memory
        # rhs = (self.hardware.on_chip_mem_bytes - const_mem) / (bw_w_bytes * C)

        # # Identify linear-type FUs
        # linear_fus = [
        #     fu for fu in fu_list
        #     if self.hardware_model.fus.get(fu, {}).get("type") == "Linear"
        # ]

        # if linear_fus:
        #     expr_mem = " + ".join(f"{fu}" for fu in linear_fus)
        #     expr_mem += f" <= {rhs}"
        #     self.ir_constraints.append(ConstraintExpr("memory_capacity", expr_mem))

        # # After building all IR constraints, write them to logger
        # if self.logger:
        #     for c in self.ir_constraints:
        #         self.logger.log_constraint(c.name, c.expr)


    # ============================================================
    # =============== IR Block Builder ============================
    # ============================================================
    def build_ir_block(self, seq_len: int):
        """Return IR block for a single seq_len (not writing to file)."""
        return {
            "seq_len": seq_len,
            "variables": [fu for fu in self.hardware_model.fus.keys()],
            "constraints": [
                {"name": c.name, "expr": c.expr}
                for c in self.ir_constraints
            ],
            "parameters": self.ir_parameters
        }

    # ============================================================
    # =============== Save Master IR ==============================
    # ============================================================
    @staticmethod
    def save_master_ir(filepath: str, master_ir_dict: dict):
        """Save all IR blocks into one combined JSON file."""
        with open(filepath, "w") as f:
            json.dump(master_ir_dict, f, indent=4)
        print(f"[Constraints] Master IR saved to {filepath}")
