#!/usr/bin/env python3
"""
Analytic framework for evaluating fully-temporal FPGA accelerators
for transformer inference.

Given three JSON configuration files (architecture, hardware, and model),
the analyzer estimates end-to-end inference latency and reports
on-chip memory, BRAM, and off-chip bandwidth feasibility.

Inputs:
    gpt2_arch_temporal.json   Architecture graph (nodes, dimensions, schedule)
    hardware_config.json      Target FPGA device parameters
    gpt2-medium_config.json   Model parameters (layers, hidden size, etc.)

Usage:
    python Temporal_Analytic_Framework.py
    python Temporal_Analytic_Framework.py --seq-lens 32,64,128
    python Temporal_Analytic_Framework.py --seq-len 32 64 128
"""

import json
import sys
import math
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

# Fallback values used only when the corresponding JSON entry is missing.
DEFAULTS = {
    "config.seq_len": 1024,
}

# Streaming-edge FIFO depth assumed by the on-chip storage model.
FIFO_DEPTH = 1

# Target device key in hardware_config.json. If None, the first device is used.
DEVICE_KEY = None

# Default weight reuse factor for the fully-temporal baseline.
DEFAULT_REUSE_FACTOR = 1

# Datatype for both weights and activations.
# Supported values: "int8", "int16", "fp16", "fp32", "fp64".
DATATYPE = "fp32"


# ============================================================
# Architectural efficiency factors
# ============================================================
# Effective MAC throughput is modeled as
#
#     Mtot_eff = Mtot_raw * eta_pe_loop_pipeline_ii * eta_compute_memory_time_ratio
#
# Each factor represents an independent architectural design choice
# and is intended to be set from the design specification, not from
# post-synthesis measurements. The analyzer remains a pre-synthesis
# prediction tool.
# ============================================================

# PE-loop pipeline initiation interval efficiency.
# Equal to 1 / II of the PE (matmul) loop. The name is qualified with
# "PE loop" because the analyzer accounts for non-linear loop II
# separately through the non-linear cycle model.
PE_LOOP_PIPELINE_INITIATION_INTERVAL = 1.0

# Compute / memory time ratio.
# Fraction of total matmul runtime spent on PE computation rather than
# on load/store stages that are not overlapped with compute.
COMPUTE_MEMORY_TIME_RATIO = 0.2


# ============================================================
# Datatype bit-widths
# ============================================================
# B_W (weight bits) and B_A (activation bits) are derived from DATATYPE
# so that the bandwidth and on-chip storage models stay consistent with
# the selected datatype.
# ============================================================

DATATYPE_BITS = {
    "int8":  8,
    "int16": 16,
    "fp16":  16,
    "fp32":  32,
    "fp64":  64,
}

if DATATYPE not in DATATYPE_BITS:
    print(f"[warn] Unknown datatype '{DATATYPE}' for bit-width, falling back to fp32 (32 bits)")
    B_W = 32
    B_A = 32
else:
    B_W = DATATYPE_BITS[DATATYPE]
    B_A = DATATYPE_BITS[DATATYPE]


# ============================================================
# Non-linear hardware throughput
# ============================================================
# Each entry specifies the synthesis-measured cycle count required to
# process `elements_per_chunk` elements for the corresponding operation.
# These values should be re-calibrated when the target hardware changes.
# ============================================================

NONLINEAR_THROUGHPUT = {
    "LayerNorm": {"cycles_per_chunk": 14348, "elements_per_chunk": 1024},
    "Softmax":   {"cycles_per_chunk": 30920, "elements_per_chunk": 1024},
    "GELU":      {"cycles_per_chunk": 15364, "elements_per_chunk": 1024},
    "Add":       {"cycles_per_chunk": 4101,  "elements_per_chunk": 1024},
}


# ============================================================
# Command-line argument parsing
# ============================================================

def parse_seq_list(argv):
    """Parse `--seq-lens 32,64,128` or `--seq-len 32 64 128` from argv."""
    vals = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--seq-lens", "--seq-len"):
            if i + 1 < len(argv) and ("," in argv[i+1]):
                vals = [int(x) for x in argv[i+1].split(",") if x.strip()]
                i += 2
                continue
            j = i + 1
            buf = []
            while j < len(argv) and not argv[j].startswith("-"):
                buf.append(argv[j])
                j += 1
            if buf:
                vals = [int(x) for x in buf]
            i = j
            continue
        i += 1
    return vals if vals else None


SEQ_LENS_OVERRIDE = parse_seq_list(sys.argv[1:])


# ============================================================
# General helpers
# ============================================================

def load_json(filename: str):
    """Load a JSON file from disk."""
    with Path(filename).open("r") as f:
        return json.load(f)


def resolve_path(root_dict: dict, path: str, fallback_key: str = None):
    """
    Resolve a dotted path (e.g., "hidden_size") within a configuration
    dictionary. If missing and a fallback_key is supplied, return the
    value from DEFAULTS.
    """
    if path in root_dict:
        return root_dict[path]
    cur = root_dict
    ok = True
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            ok = False
            break
    if ok:
        return cur
    if fallback_key and fallback_key in DEFAULTS:
        val = DEFAULTS[fallback_key]
        print(f"[warn] Missing '{path}', using default for '{fallback_key}': {val}")
        return val
    raise KeyError(f"Cannot resolve path '{path}' in provided JSONs.")


def build_symbol_table(dimensions_map: dict, model_cfg: dict, runtime_cfg: dict = None):
    """
    Build a symbol table from the architecture dimensions map.

    A dimensions map looks like
        {
            "D_MODEL": "model.hidden_size",
            "D_FFN":   "model.ffn_size",
            "SEQ_LEN": "config.seq_len"
        }
    where 'model.*' resolves against model_cfg and 'config.*' resolves
    against runtime_cfg (falling back to DEFAULTS when absent).
    """
    symbols = {}
    for sym, ref in dimensions_map.items():
        if not isinstance(ref, str):
            raise ValueError(f"Dimension '{sym}' must map to a string path, got: {ref}")
        if ref.startswith("model."):
            path = ref.split("model.", 1)[1]
            symbols[sym] = resolve_path(model_cfg, path)
        elif ref.startswith("config."):
            path = ref.split("config.", 1)[1]
            if runtime_cfg is None:
                symbols[sym] = DEFAULTS.get(ref, DEFAULTS.get(f"config.{path}"))
                if symbols[sym] is None:
                    raise KeyError(f"'{ref}' not provided and no default set.")
                print(f"[warn] No runtime config, using default for '{ref}': {symbols[sym]}")
            else:
                symbols[sym] = resolve_path(runtime_cfg, path, fallback_key=ref)
        else:
            try:
                symbols[sym] = int(ref)
            except Exception:
                raise ValueError(f"Unsupported dimension ref '{ref}' for symbol '{sym}'")
    return symbols


# ============================================================
# Symbolic expression helpers
# ============================================================

def token_to_sym(token):
    """Map a dimension token to its display symbol."""
    if isinstance(token, int):
        return str(token)

    mapping = {
        "SEQ_LEN":            "l",
        "D_MODEL":            "d",
        "D_FFN":              "f",
        "D_HEAD":             "h",
        "NUM_HEADS":          "H",
        "NUM_KV_HEADS":       "Hkv",
        "WINDOW_SIZE":        "w",
        "NUM_EXPERTS":        "E",
        "NUM_ACTIVE_EXPERTS": "Ea",
    }
    return mapping.get(token, str(token))


def mac_expr_for_node(node: dict):
    """Build a human-readable symbolic MAC expression for a node."""
    mc = node.get("mac_calc", {})
    if not mc:
        return None

    parts = []
    for v in mc.values():
        if isinstance(v, int):
            parts.append(str(v))
        elif isinstance(v, str):
            sub_tokens = [tok.strip() for tok in v.split("*") if tok.strip()]
            if not sub_tokens:
                continue
            pretty_subs = [token_to_sym(tok) for tok in sub_tokens]
            parts.append("*".join(pretty_subs))
        else:
            raise TypeError(f"Unsupported mac_calc value type in expr: {type(v)} ({v})")

    return "*".join(parts) if parts else None


def eval_dim_expr(expr, symbols: dict) -> int:
    """
    Evaluate a dimension expression. The expression can be an integer,
    a symbol name, or a multiplication-only string such as "A * B * 4".
    """
    if isinstance(expr, int):
        return expr

    if isinstance(expr, str):
        factors = [tok.strip() for tok in expr.split("*") if tok.strip()]
        if not factors:
            raise ValueError(f"Empty dimension expression: {expr}")
        val = 1
        for fac in factors:
            val *= dim_value(fac, symbols)
        return int(val)

    raise TypeError(f"Unsupported mac_calc value type: {type(expr)} ({expr})")


def dim_value(token: str, symbols: dict):
    """Resolve a single token to its numeric value."""
    if isinstance(token, int):
        return token
    if token in symbols:
        return symbols[token]
    try:
        return int(token)
    except Exception:
        raise KeyError(f"Unknown dimension token '{token}'. Known symbols: {list(symbols.keys())}")


def macs_for_node(node: dict, symbols: dict) -> int:
    """Total MACs for a node, computed as the product of all mac_calc dimensions."""
    mc = node.get("mac_calc")
    if not mc:
        return 0

    macs = 1
    for v in mc.values():
        macs *= eval_dim_expr(v, symbols)
    return int(macs)


def macs_for_linear_node(node: dict, symbols: dict) -> int:
    """Return MACs only for Linear nodes; zero otherwise."""
    if node.get("type") != "Linear":
        return 0
    return macs_for_node(node, symbols)


# ============================================================
# Bandwidth model helpers
# ============================================================

def max_linear_macs_in_layer(nodes, symbols):
    """
    Return (max_macs, node_name) for the largest Linear node in the layer.
    Used to identify the bottleneck Linear node by name.
    """
    best = 0
    best_name = None
    for n in nodes:
        if n.get("type") == "Linear":
            val = macs_for_node(n, symbols)
            if val > best:
                best = val
                best_name = n.get("name", "unknown_linear")
    return best, best_name


def auto_node_reuse_factor(node: dict, symbols: dict):
    """
    Estimate the weight reuse factor for a node from its JSON dimensions
    under a weight-stationary assumption.

    Returns the estimated reuse factor, or None when no estimation is
    possible (missing mac_calc or required dimensions).
    """
    t = node.get("type")
    mc = node.get("mac_calc") or {}

    if not mc:
        return None

    # Linear: MAC = l * d_in * d_out; weights are (d_in, d_out), reuse ~ l.
    if t == "Linear":
        l_expr = mc.get("l")
        if l_expr is None:
            return None
        l_val = eval_dim_expr(l_expr, symbols)
        return max(1, int(l_val))

    # BatchMatmul: MAC = m * k * n * b; weights are (k, n), reuse ~ m * b.
    if t == "BatchMatmul":
        m_expr = mc.get("m")
        if m_expr is None:
            return None
        m_val = eval_dim_expr(m_expr, symbols)

        b_expr = mc.get("b")
        if b_expr is not None:
            b_val = eval_dim_expr(b_expr, symbols)
        else:
            b_val = 1

        reuse = m_val * b_val
        return max(1, int(reuse))

    return None


def node_reuse_factor(node: dict, symbols: dict) -> int:
    """Wrapper around auto_node_reuse_factor that always returns >= 1."""
    r = auto_node_reuse_factor(node, symbols)
    if r is None:
        return DEFAULT_REUSE_FACTOR
    return max(1, int(r))


def bw_req_bytes_per_sec_for_device(Mtot: int, freq_hz: int, b_w_bits: int, reuse_factor: int) -> float:
    """
    Off-chip bandwidth required to keep all MAC units busy:

        B_req = b_w * (Mtot / r) * f   [bits/s]  -> bytes/s
    """
    eff_weights_per_cycle = max(1, int(Mtot) // max(1, int(reuse_factor)))
    bw_bits_per_s = int(b_w_bits) * eff_weights_per_cycle * int(freq_hz)
    return bw_bits_per_s / 8.0


# ============================================================
# Display helpers
# ============================================================

def human_bits(x):
    for unit in ["b", "Kb", "Mb", "Gb", "Tb"]:
        if x < 1024:
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} Pb"


def human_bytes(x):
    for unit in ["B/s", "KB/s", "MB/s", "GB/s", "TB/s"]:
        if x < 1024:
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} PB/s"


def human_bytes_decimal(x):
    if x < 1e3:
        return f"{x:.2f} B/s"
    elif x < 1e6:
        return f"{x/1e3:.2f} kB/s"
    elif x < 1e9:
        return f"{x/1e6:.2f} MB/s"
    else:
        return f"{x/1e9:.2f} GB/s"


# ============================================================
# Non-linear shape inference and cycle calculation
# ============================================================

def find_node_by_name(name, all_nodes):
    """Look up a node by its name within the layer graph."""
    for n in all_nodes:
        if n.get("name") == name:
            return n
    return None


def infer_output_shape(node, all_nodes, symbols):
    """
    Return the (rows, cols) output shape of a node.

    Rules:
        - Linear:                       (l, d_out)
        - BatchMatmul:                  (m, n)
        - LayerNorm/Softmax/Add/GELU:   inherited recursively from the first input
        - source 'x_layer_in':          (SEQ_LEN, D_MODEL)
    """
    node_type = node.get("type")
    SEQ_LEN = symbols.get("SEQ_LEN", 1)
    D_MODEL = symbols.get("D_MODEL", 1)

    if node_type == "Linear":
        mc = node.get("mac_calc", {})
        l_expr = mc.get("l", "SEQ_LEN")
        d_out_expr = mc.get("d_out", "D_MODEL")
        l = eval_dim_expr(l_expr, symbols)
        d_out = eval_dim_expr(d_out_expr, symbols)
        return (l, d_out)

    if node_type == "BatchMatmul":
        mc = node.get("mac_calc", {})
        m_expr = mc.get("m", "SEQ_LEN")
        n_expr = mc.get("n", "SEQ_LEN")
        m = eval_dim_expr(m_expr, symbols)
        n = eval_dim_expr(n_expr, symbols)
        return (m, n)

    if node_type in ("LayerNorm", "Softmax", "Add", "GELU"):
        inputs = node.get("inputs", [])
        if not inputs:
            return (SEQ_LEN, D_MODEL)
        first_src = inputs[0].get("source")
        if first_src == "x_layer_in":
            return (SEQ_LEN, D_MODEL)
        parent = find_node_by_name(first_src, all_nodes)
        if parent is None:
            return (SEQ_LEN, D_MODEL)
        return infer_output_shape(parent, all_nodes, symbols)

    return (SEQ_LEN, D_MODEL)


def nonlinear_op_type(node):
    """
    Return the non-linear operation type for a node:
        - if the node has an 'activation' entry registered in
          NONLINEAR_THROUGHPUT, return that activation
        - else if the node 'type' is registered, return that type
        - otherwise None
    """
    activation = node.get("activation")
    if activation in NONLINEAR_THROUGHPUT:
        return activation
    node_type = node.get("type")
    if node_type in NONLINEAR_THROUGHPUT:
        return node_type
    return None


def nonlinear_elements(node, all_nodes, symbols, model_cfg):
    """
    Total number of elements processed by a non-linear node per layer.
    Softmax additionally scales by the number of attention heads.
    """
    op_type = nonlinear_op_type(node)
    if op_type is None:
        return 0

    rows, cols = infer_output_shape(node, all_nodes, symbols)
    elements = rows * cols

    if op_type == "Softmax":
        num_heads = model_cfg.get("num_attention_heads", 1)
        elements *= num_heads

    return elements


def calculate_nonlinear_cycles(nodes, symbols, model_cfg, verbose=False):
    """Sum cycles across all non-linear nodes in a single layer."""
    total_cycles = 0
    breakdown = {}

    for node in nodes:
        op_type = nonlinear_op_type(node)
        if op_type is None:
            continue

        elements = nonlinear_elements(node, nodes, symbols, model_cfg)
        if elements == 0:
            continue

        spec = NONLINEAR_THROUGHPUT[op_type]
        n_chunks = math.ceil(elements / spec["elements_per_chunk"])
        node_cycles = n_chunks * spec["cycles_per_chunk"]
        total_cycles += node_cycles

        if verbose:
            name = node.get("name", "?")
            breakdown[name] = {
                "op": op_type,
                "elements": elements,
                "chunks": n_chunks,
                "cycles": node_cycles,
            }

    if verbose:
        return total_cycles, breakdown
    return total_cycles


# ============================================================
# Load configurations
# ============================================================

arch = load_json("gpt2_arch_temporal.json")
hw_all = load_json("hardware_config.json")
model_cfg = load_json("gpt2-medium_config.json")

# Optional runtime override; DEFAULTS are used if not present.
runtime_cfg_path = Path("runtime_config.json")
runtime_cfg = None
if runtime_cfg_path.exists():
    runtime_cfg = load_json(str(runtime_cfg_path))

# Select target device.
DEVICE_KEY = next(iter(hw_all.keys())) if DEVICE_KEY is None else DEVICE_KEY
hw = hw_all[DEVICE_KEY]

# Hardware parameters.
Mtot_raw = hw["total_macs"]
freq = hw["frequency_hz"]
on_chip_bytes = hw["on_chip_mem_bytes"]
on_chip_bits = on_chip_bytes * 8
num_bram_blocks = hw["num_bram_blocks"]
offchip_bw_Bps = hw["off_chip_mem_bw_bytes_per_sec"]
S_BRAM_bits = on_chip_bits // num_bram_blocks


# ============================================================
# Effective Mtot
#     Mtot_eff = Mtot_raw * eta_pe_loop_pipeline_ii * eta_compute_memory_time_ratio
# ============================================================

combined_eff = PE_LOOP_PIPELINE_INITIATION_INTERVAL * COMPUTE_MEMORY_TIME_RATIO
Mtot = Mtot_raw * combined_eff

# Build the symbol table from the architecture dimensions.
dimensions_map = arch.get("dimensions", {})
symbols = build_symbol_table(dimensions_map, model_cfg, runtime_cfg)

# Number of transformer layers in the target model.
C_LAYERS = model_cfg.get("num_layers", 1)

# Sequence-length sweep: CLI override takes precedence over the symbol table.
seq_list = SEQ_LENS_OVERRIDE if SEQ_LENS_OVERRIDE else [symbols.get("SEQ_LEN", DEFAULTS["config.seq_len"])]


# ============================================================
# Header summary
# ============================================================

print("=== Fully Temporal Latency & Constraints (JSON-driven) ===")
print(f"Model: {model_cfg.get('model_name','unknown')} | Layers: {C_LAYERS}")
print(f"Resolved dimensions: {symbols}")
print(f"Device: {hw['name']} | f={freq} Hz | OnChip={human_bits(on_chip_bits)} | BRAM blocks={num_bram_blocks}")
print(f"Datatype: {DATATYPE} | bit-width: {B_W} bits (weight) / {B_A} bits (activation)")
print(f"Efficiency factors (user input):")
print(f"  - PE loop pipeline II efficiency    : {PE_LOOP_PIPELINE_INITIATION_INTERVAL:.4f}")
print(f"  - Compute / memory time ratio       : {COMPUTE_MEMORY_TIME_RATIO:.4f}")
print(f"  - Combined Mtot multiplier          : {combined_eff:.6f}")
print(f"Mtot: {Mtot_raw} raw MAC -> {Mtot:.2f} effective MAC/cycle")
print(f"Off-chip BW: {human_bytes_decimal(offchip_bw_Bps)} -> {offchip_bw_Bps * 8 / B_W / 1e9:.2f} G weights/s @ {B_W}-bit")
print(f"Non-linear throughput:")
for op, spec in NONLINEAR_THROUGHPUT.items():
    print(f"  {op:>10}: {spec['cycles_per_chunk']:>6} cycles / {spec['elements_per_chunk']} elements")
print()

header = (
    f"{'SEQ':>5} | {'Matmul(ms)':>11} | {'Nonlin(ms)':>11} | {'Total(ms)':>11} | "
    f"{'SRAM OK':>7} | {'BRAM OK':>7} | "
    f"{'BW OK':>6} | {'BW req (peak)':>16} | {'BW avail':>10} | {'Mi node':>12}"
)


# ============================================================
# Symbolic latency model (printed for documentation)
# ============================================================

nodes = arch["layer_graph"]["nodes"]

layer_mac_expr_terms = []
for n in nodes:
    expr = mac_expr_for_node(n)
    if expr is not None:
        layer_mac_expr_terms.append(f"{n.get('name','?')}:{expr}")

raw_layer_mac_expr = " + ".join(
    e.split(":", 1)[1] for e in layer_mac_expr_terms
) if layer_mac_expr_terms else "0"

print("\n[Symbolic latency model]")
print("MACs_per_layer(l,d,f)   = " + raw_layer_mac_expr)
print("Total_MACs(l,d,f,C)     = C * MACs_per_layer(l,d,f)")
print("Mtot_eff                = Mtot_raw * eta_pe_loop_pipeline_ii * eta_compute_memory_time_ratio")
print("Compute_bound_MACrate   = Mtot_eff * freq")
print("BW_bound_MACrate        = (offchip_bw_Bps * 8 / B_W) * reuse_r")
print("Effective_MACrate       = min(Compute_bound_MACrate, BW_bound_MACrate)")
print("Matmul_Latency(l)       = Total_MACs(l,d,f,C) / Effective_MACrate")
print("Nonlin_Latency(l)       = sum(ceil(elements/chunk_size) * cycles_per_chunk) / freq * C")
print("Total_Latency(l)        = Matmul_Latency + Nonlin_Latency\n")


# ============================================================
# Symbolic SRAM / BRAM / bandwidth constraints (printed)
# ============================================================

kv_nodes = [n for n in nodes if n.get('caches_output') == 'kv_cache']
num_kv = len(kv_nodes)

need_xact = any(n.get('requires_input_buffer', False) for n in nodes)

streaming_edges = 0
for n in nodes:
    for inp in n.get("inputs", []):
        if inp.get("type") == "streaming":
            streaming_edges += 1

print("[Symbolic SRAM / BRAM constraints]")
print("Let:")
print("  N_kv           = number of nodes with caches_output == 'kv_cache'")
print("  N_stream_edges = number of streaming edges in layer_graph")
print("  has_Xact       = any node has requires_input_buffer == True")
print()
print("S_tile_bits        = 2 * Mtot * B_W")
print("S_KV_bits(l)       = N_kv * 2 * l * D_MODEL * B_A")
print("S_Xact_bits(l)     = has_Xact ? (l * D_MODEL * B_A) : 0")
print("S_FIFO_bits        = N_stream_edges * FIFO_DEPTH * B_A")
print("SRAM_needed_bits(l)= S_tile_bits + S_KV_bits(l) + S_Xact_bits(l) + S_FIFO_bits")
print("On_chip_bits       = on_chip_mem_bytes * 8")
print("SRAM_ok(l) <=> SRAM_needed_bits(l) <= On_chip_bits")
print()
print("S_BRAM_bits        = On_chip_bits / num_bram_blocks")
print("BRAM_blocks_needed(l) ~= SRAM_needed_bits(l) / S_BRAM_bits\n")

print("[Symbolic BW constraint]")
print("BW_req_bytes_per_s   = (B_W * (Mtot / reuse_r) * freq) / 8")
print("BW_avail_bytes_per_s = offchip_bw_Bps")
print("BW_ok <=> BW_req_bytes_per_s <= BW_avail_bytes_per_s\n")


print(header)
print("-" * len(header))


# ============================================================
# Per-sequence-length numerical evaluation
# ============================================================

for lval in seq_list:
    local_symbols = dict(symbols)
    local_symbols["SEQ_LEN"] = lval

    # MACs per layer and total MACs across all layers.
    macs_per_layer = sum(macs_for_node(n, local_symbols) for n in nodes)
    total_macs_all_layers = macs_per_layer * C_LAYERS

    # Determine the dominant reuse factor from the largest Linear / BatchMatmul node.
    reuse_r = DEFAULT_REUSE_FACTOR
    best_macs = -1

    for n in nodes:
        if n.get("type") not in ("Linear", "BatchMatmul"):
            continue

        macs_n = macs_for_node(n, local_symbols)
        if macs_n <= 0:
            continue

        r_i = auto_node_reuse_factor(n, local_symbols)
        if r_i is None or r_i <= 0:
            continue

        if macs_n > best_macs:
            best_macs = macs_n
            reuse_r = int(r_i)

    # Compute-bound and bandwidth-bound MAC rates.
    compute_limited_macs_per_sec = Mtot * freq

    if offchip_bw_Bps > 0:
        weights_per_sec = offchip_bw_Bps * 8.0 / B_W
        bw_limited_macs_per_sec = weights_per_sec * reuse_r
    else:
        bw_limited_macs_per_sec = 0.0

    if bw_limited_macs_per_sec <= 0:
        effective_macs_per_sec = compute_limited_macs_per_sec
    else:
        effective_macs_per_sec = min(compute_limited_macs_per_sec,
                                     bw_limited_macs_per_sec)

    # Matmul latency.
    matmul_latency_sec = total_macs_all_layers / effective_macs_per_sec
    matmul_latency_ms = matmul_latency_sec * 1e3

    # Non-linear latency.
    nonlinear_cycles_per_layer = calculate_nonlinear_cycles(nodes, local_symbols, model_cfg)
    nonlinear_cycles_total = nonlinear_cycles_per_layer * C_LAYERS
    nonlinear_latency_sec = nonlinear_cycles_total / freq
    nonlinear_latency_ms = nonlinear_latency_sec * 1e3

    # End-to-end latency.
    total_latency_ms = matmul_latency_ms + nonlinear_latency_ms

    # On-chip storage breakdown.
    kv_nodes = [n for n in nodes if n.get("caches_output") == "kv_cache"]
    D_MODEL = int(local_symbols["D_MODEL"])
    S_KV_bits = len(kv_nodes) * 2 * int(lval) * D_MODEL * B_A

    need_xact = any(n.get("requires_input_buffer", False) for n in nodes)
    S_Xact_bits = (int(lval) * D_MODEL * B_A) if need_xact else 0

    streaming_edges = 0
    for n in nodes:
        for inp in n.get("inputs", []):
            if inp.get("type") == "streaming":
                streaming_edges += 1
    S_FIFO_bits = streaming_edges * FIFO_DEPTH * B_A

    S_tile_bits_peak = 2 * Mtot * B_W

    SRAM_needed_bits_peak = S_tile_bits_peak + S_KV_bits + S_Xact_bits + S_FIFO_bits
    SRAM_ok = SRAM_needed_bits_peak <= on_chip_bits

    # BRAM block-count estimate.
    R_i_blocks  = S_tile_bits_peak / S_BRAM_bits
    KV_blocks   = S_KV_bits        / S_BRAM_bits
    Xact_blocks = S_Xact_bits      / S_BRAM_bits
    FIFO_blocks = S_FIFO_bits      / S_BRAM_bits
    BRAM_blocks_needed_peak = R_i_blocks + KV_blocks + Xact_blocks + FIFO_blocks
    BRAM_ok = BRAM_blocks_needed_peak <= num_bram_blocks

    # Bandwidth feasibility.
    Mi, mi_node_name = max_linear_macs_in_layer(nodes, local_symbols)
    peak_bw_Bps = bw_req_bytes_per_sec_for_device(Mtot, freq, B_W, reuse_r)
    BW_ok = peak_bw_Bps <= offchip_bw_Bps

    print(
        f"{lval:>5} | {matmul_latency_ms:>11.3f} | {nonlinear_latency_ms:>11.3f} | {total_latency_ms:>11.3f} | "
        f"{'OK' if SRAM_ok else 'NG':>7} | {'OK' if BRAM_ok else 'NG':>7} | "
        f"{'OK' if BW_ok else 'NG':>6} | "
        f"{human_bytes_decimal(peak_bw_Bps):>16} | {human_bytes_decimal(offchip_bw_Bps):>10} | {mi_node_name or '-':>12}"
    )


# ============================================================
# Per-node non-linear breakdown for the largest SEQ
# ============================================================

print("\n[Non-linear breakdown for SEQ = {}]".format(seq_list[-1]))
local_symbols = dict(symbols)
local_symbols["SEQ_LEN"] = seq_list[-1]
_, breakdown = calculate_nonlinear_cycles(nodes, local_symbols, model_cfg, verbose=True)
print(f"{'Node':>20} | {'Op':>10} | {'Elements':>12} | {'Chunks':>8} | {'Cycles':>12}")
print("-" * 75)
for name, info in breakdown.items():
    print(f"{name:>20} | {info['op']:>10} | {info['elements']:>12,} | {info['chunks']:>8,} | {info['cycles']:>12,}")