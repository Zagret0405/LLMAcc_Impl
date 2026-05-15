import json
from collections import defaultdict
from pyomo.environ import (
    ConcreteModel, Var, Objective,
    Constraint, SolverFactory, NonNegativeReals, value, minimize
)

from latency_calculation import get_latency_function
from FU_mapping import HardwareModel
from hardware import HardwareDevice


class ConstraintSolver:
    """
    Dataflow-style solver (teammate-style structure).
    Stage-based pipeline balancing:
        minimize T
        s.t. stage_MACs <= T * stage_M

    Fixes in this version:
      - Distinguish node names vs FU names via fu_map
      - Stage balancing uses node->FU aggregated MACs
      - Zero-MAC enforcement works even when node != FU
      - Do NOT skip bandwidth constraint
      - Only skip memory_capacity (same as teammate now)
      - Auto-build stages by topological depth (works for GPT-OSS)
    """

    def __init__(self, hardware_model: HardwareModel,
                 hardware_device: HardwareDevice,
                 constraint_path: str, seq_len: int,
                 logger=None):

        self.hardware_model = hardware_model
        self.hardware_device = hardware_device
        self.constraint_path = constraint_path
        self.seq_len = str(seq_len)
        self.logger = logger

        # Load constraint.json
        with open(constraint_path, "r") as f:
            all_constraints = json.load(f)

        if self.seq_len not in all_constraints:
            raise ValueError(f"Seq_len={self.seq_len} not found")

        self.constraint_set = all_constraints[self.seq_len]
        self.variables = self.constraint_set["variables"]        # FU variable list
        self.constraints = self.constraint_set["constraints"]

        self.params = self.constraint_set.get("parameters", {})
        self.num_layers = self.params.get("num_layers", 1)
        self.layers_per_fpga = self.params.get("layers_per_fpga", 1)

        # Latency function for final evaluation
        self.latency_fn = get_latency_function(self.hardware_model, logger=self.logger)
        self.model = None
        print("[DEBUG] deps of attn_norm =",
              self.hardware_model.layer_graph.dependencies.get("attn_norm", []))

    # ---------------------------------------------------------
    # Helper: aggregate node MACs to FU MACs
    # ---------------------------------------------------------
    def _aggregate_node_macs_to_fu(self):
        """Return dict: FU_name -> total_MACs_of_nodes_mapped_to_it."""
        node_macs = self.hardware_model.layer_graph.node_macs
        fu_map = self.hardware_model.fu_map

        fu_macs = {fu: 0.0 for fu in self.variables}
        for node_name, mac in node_macs.items():
            fu_name = fu_map.get(node_name)
            if fu_name in fu_macs:
                fu_macs[fu_name] += mac
        return fu_macs

    # ---------------------------------------------------------
    # NEW: Auto-build stages by topological depth
    # ---------------------------------------------------------
    def _compute_topological_stages(self):
        """
        Automatically group nodes into stages according to topological order.
        Parent format can be:
        deps[node] = [ {"source": "...", "type": "streaming"}, ... ]
        """
        layer_graph = self.hardware_model.layer_graph
        deps = layer_graph.dependencies
        sorted_nodes = layer_graph.sorted_nodes  # list of node names (str)

        stage_id = {}

        for node in sorted_nodes:
            parents = deps.get(node, [])
            # parents is list of dict: {"source": ..., "type": ...}
            parent_names = [p["source"] for p in parents if "source" in p]

            if not parent_names:
                stage_id[node] = 0
            else:
                stage_id[node] = max(stage_id[parent] for parent in parent_names) + 1

        # Group nodes by stage id
        stages = {}
        for node, sid in stage_id.items():
            stages.setdefault(sid, []).append(node)

        print("[DEBUG] Auto stages:", stages)
        return stages

    # ---------------------------------------------------------
    # Build Pyomo model (DATAFLOW VERSION)
    # ---------------------------------------------------------
    def build_model(self):
        model = ConcreteModel()

        # =============================
        # 1) Create M vars (FU-level)
        # =============================
        M_list = self.variables
        model.M_vars = Var(M_list, domain=NonNegativeReals)

        # =============================
        # 2) Add JSON constraints (skip only memory)
        # =============================
        for c in self.constraints:
            name = c["name"]

            # Only skip memory_capacity; DO NOT skip BW now
            if name in ["memory_capacity"]:
                print(f"[INFO] Skipping hard constraint in solver: {name}")
                continue

            expr_str = c["expr"]
            env = {v: model.M_vars[v] for v in M_list}
            expr = eval(expr_str, {"__builtins__": {}}, env)
            model.add_component(name, Constraint(expr=expr))

            if self.logger:
                self.logger.log_constraint(name, expr_str)

            print("[DEBUG] Constraint:", name, expr_str)

        # =============================
        # 3) Stage definition (auto from DAG)
        # =============================
        # Instead of hard-coded attn_qkv/attn_sdp/..., we use topo depth
        stages = self._compute_topological_stages()

        node_macs = self.hardware_model.layer_graph.node_macs
        fu_map = self.hardware_model.fu_map

        # =============================
        # 4) Force zero-MAC FU to M=0
        #    (based on aggregated FU MACs)
        # =============================
        fu_macs = self._aggregate_node_macs_to_fu()
        for fu, mac in fu_macs.items():
            if fu in M_list and mac == 0:
                cname = f"zero_mac_{fu}"
                model.add_component(cname, Constraint(expr=model.M_vars[fu] == 0))
                print(f"[INFO] Force zero-MAC FU {fu} to M=0")

        # =============================
        # 5) Stage-balance constraints
        #    (aggregate stage MACs to FU)
        # =============================
        model.T = Var(domain=NonNegativeReals)

        for sid, node_list in stages.items():

            # Aggregate MAC per FU inside this stage
            stage_fu_macs = defaultdict(float)
            for node_name in node_list:
                mac = node_macs.get(node_name, 0)
                if mac <= 0:
                    continue
                fu = fu_map.get(node_name)
                if fu in M_list:
                    stage_fu_macs[fu] += mac

            if not stage_fu_macs:
                continue

            mac_sum = sum(stage_fu_macs.values())
            M_s = sum(model.M_vars[fu] for fu in stage_fu_macs.keys())

            cname = f"stage_balance_{sid}"
            model.add_component(
                cname,
                Constraint(expr=mac_sum <= model.T * M_s)
            )

            if self.logger:
                self.logger.log_constraint(
                    cname,
                    f"{mac_sum} <= T * sum(M[{list(stage_fu_macs.keys())}])"
                )

            print(f"[DEBUG] Added stage constraint for stage {sid}: "
                  f"MAC={mac_sum}, FUs={list(stage_fu_macs.keys())}")

        # =============================
        # 6) Objective = minimize T
        # =============================
        model.OBJ = Objective(expr=model.T, sense=minimize)

        self.model = model
        return model

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------
    def solve(self, solver_name="ipopt"):
        if self.model is None:
            self.build_model()

        solver = SolverFactory(solver_name)
        res = solver.solve(self.model, tee=False)

        if self.logger:
            self.logger.log_solver_status(
                res.solver.status,
                res.solver.termination_condition
            )

        sol = {}
        for v in self.variables:
            raw = float(value(self.model.M_vars[v]))
            if abs(raw) < 1e-7:
                raw = 0.0
            if raw < 0:
                raw = 0.0
            sol[v] = raw

        return sol

    # ---------------------------------------------------------
    # Final latency evaluation using real DAG
    # ---------------------------------------------------------
    def evaluate_latency(self, M_dict):
        single_cycles = self.latency_fn(M_dict)

        N = self.num_layers
        C = self.layers_per_fpga if self.layers_per_fpga > 0 else 1
        total_cycles = single_cycles * (N / C)

        seconds = total_cycles / self.hardware_device.frequency_hz
        return total_cycles, seconds

    # ---------------------------------------------------------
    # Full pipeline
    # ---------------------------------------------------------
    def run(self):
        if self.logger:
            self.logger.log_separator(self.seq_len)

        solution = self.solve()
        cycles, seconds = self.evaluate_latency(solution)

        if self.logger:
            for fu, m in solution.items():
                self.logger.log_fu_allocation(fu, m)
            self.logger.log_latency(seconds)

        return {
            "solution": solution,
            "latency_cycles": cycles,
            "latency_seconds": seconds
        }
