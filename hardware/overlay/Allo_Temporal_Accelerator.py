"""
Allo Temporal Accelerator

A TPU-style temporal accelerator for transformer inference, implemented
in the Allo HLS framework and targeting the Xilinx Alveo U280. The
accelerator consists of an M x M PE matrix-multiply engine and a vector
processing unit (VPU) covering all non-linear operations (vector add,
scale, ReLU, GELU, layer normalization, exp, and softmax). The host
issues a program of micro-instructions through a shared on-chip memory
and a small instruction memory.

This file contains the kernel definitions, the schedule, and a
top-level entry point that builds the accelerator and runs Vitis HLS
hardware emulation.
"""

import allo
from allo import Memory
from allo.ir.types import int32, float32, stateful
import numpy as np
import os
import shutil


# ============================================================
# TPU configuration
# ============================================================

class TPUConfig:
    """Static configuration parameters of the temporal accelerator."""

    M: int = 32

    @property
    def MEM_SIZE(self) -> int:
        return self.M * self.M * 8

    @property
    def IMEM_SIZE(self) -> int:
        return 32

    @property
    def BW(self) -> int:
        return 16

cfg = TPUConfig()
M, MEM_SIZE, IMEM_SIZE, BW = cfg.M, cfg.MEM_SIZE, cfg.IMEM_SIZE, cfg.BW


# ============================================================
# Host control commands
# ============================================================

CTRL_MEMCPY_H2D, CTRL_MEMCPY_D2H = 0, 1
CTRL_RUN_PROG, CTRL_MEMCPY_PROG, CTRL_LOAD_LUT = 3, 4, 5


# ============================================================
# Instruction opcodes
# ============================================================

OP_MM, OP_VADD, OP_SCALE, OP_RELU = 2, 3, 4, 6
OP_END, OP_MM_ACC, OP_GELU, OP_LN = 5, 7, 8, 9
OP_EXP = 10
OP_SOFTMAX = 11


# ============================================================
# Compute kernels
# ============================================================

def matmul(mem: float32[MEM_SIZE], addr1: int32, addr2: int32, addr3: int32, accum: int32):
    """Tile matrix multiply C = A x B (or C += A x B when accum == 1)."""
    local_A: float32[M, M]
    local_B: float32[M, M]
    local_C: float32[M, M]

    for i, j in allo.grid(M, M, name="load_A"):
        local_A[i, j] = mem[addr1 + i * M + j]
    for i, j in allo.grid(M, M, name="load_B"):
        local_B[i, j] = mem[addr2 + i * M + j]

    for i, j in allo.grid(M, M, name="load_C"):
        if accum == 1:
            local_C[i, j] = mem[addr3 + i * M + j]
        else:
            local_C[i, j] = 0.0

    for m, n in allo.grid(M, M, name="PE"):
        c: float32 = local_C[m, n]
        for k in range(M):  # Inner reduction is kept as a plain range loop.
            a: float32 = local_A[m, k]
            b: float32 = local_B[k, n]
            c = c + a * b
        local_C[m, n] = c

    for i, j in allo.grid(M, M, name="store_C"):
        mem[addr3 + i * M + j] = local_C[i, j]


def vadd(mem: float32[MEM_SIZE], addr1: int32, addr2: int32, addr3: int32):
    """Element-wise vector add over M*M elements."""
    for offset in allo.grid(M * M, name="vadd_loop"):
        mem[addr3 + offset] = mem[addr1 + offset] + mem[addr2 + offset]


def scale(mem: float32[MEM_SIZE], addr1: int32, s: float32, addr3: int32):
    """Element-wise scalar multiply."""
    for offset in allo.grid(M * M, name="scale_loop"):
        mem[addr3 + offset] = mem[addr1 + offset] * s


def relu(mem: float32[MEM_SIZE], addr1: int32, addr3: int32):
    """Element-wise ReLU activation."""
    for offset in allo.grid(M * M, name="relu_loop"):
        val: float32 = mem[addr1 + offset]
        if val > 0.0: mem[addr3 + offset] = val
        else: mem[addr3 + offset] = 0.0


def gelu(mem: float32[MEM_SIZE], addr1: int32, addr3: int32, lut: float32[1536]):
    """GELU activation implemented via a lookup table over [-4, 4]."""
    for offset in allo.grid(M * M, name="gelu_loop"):
        x: float32 = mem[addr1 + offset]
        if x <= -4.0:
            mem[addr3 + offset] = 0.0
        elif x >= 4.0:
            mem[addr3 + offset] = x
        else:
            idx: int32 = (x + 4.0) * 64.0
            if idx > 511: idx = 511
            if idx < 0: idx = 0
            mem[addr3 + offset] = lut[idx + 1024]


def layer_norm(mem: float32[MEM_SIZE], addr1: int32, addr3: int32, lut: float32[1536]):
    """Layer normalization over M*M elements using a 1/sqrt(var) lookup table."""
    sum_val: float32 = 0.0
    sq_sum_val: float32 = 0.0

    for i_sum in allo.grid(M * M, name="ln_sum"):
        val: float32 = mem[addr1 + i_sum]
        sum_val += val
        sq_sum_val += val * val

    mean: float32 = sum_val / 256.0
    var: float32 = (sq_sum_val / 256.0) - (mean * mean)

    idx: int32 = var * 100.0
    if idx > 511: idx = 511
    if idx < 0: idx = 0
    inv_std: float32 = lut[idx + 512]

    for i_norm in allo.grid(M * M, name="ln_norm"):
        val_tmp: float32 = mem[addr1 + i_norm]
        mem[addr3 + i_norm] = (val_tmp - mean) * inv_std / 1024.0


def my_exp(mem: float32[MEM_SIZE], addr_in: int32, addr_out: int32, lut: float32[1536]):
    """Element-wise exponential implemented via a lookup table over [-10, 0]."""
    for offset in allo.grid(M * M, name="exp_loop"):
        x: float32 = mem[addr_in + offset]
        if x < -10.0: x = -10.0
        if x > 0.0: x = 0.0

        idx: int32 = (x + 10.0) * 51.2
        if idx > 511: idx = 511
        if idx < 0: idx = 0
        mem[addr_out + offset] = lut[idx]


def softmax(mem: float32[MEM_SIZE], addr_in: int32, addr_tmp: int32, addr_out: int32, lut: float32[1536]):
    """Row-wise softmax over an M x M tile, using max-subtraction and the exp LUT."""
    for i1 in allo.grid(M, name="sm_outer1"):
        max_val: float32 = -999999.0
        for j_max in allo.grid(M, name="sm_find_max"):
            val: float32 = mem[addr_in + i1 * M + j_max]
            if val > max_val: max_val = val

        for j_sub in allo.grid(M, name="sm_sub"):
            mem[addr_tmp + i1 * M + j_sub] = mem[addr_in + i1 * M + j_sub] - max_val

    my_exp(mem, addr_tmp, addr_out, lut)

    for i2 in allo.grid(M, name="sm_outer2"):
        sum_exp: float32 = 0.0
        for j_sum in allo.grid(M, name="sm_sum"):
            sum_exp = sum_exp + mem[addr_out + i2 * M + j_sum]

        for j_div in allo.grid(M, name="sm_div"):
            mem[addr_out + i2 * M + j_div] = mem[addr_out + i2 * M + j_div] / sum_exp


def vpu(op: int32, mem: float32[MEM_SIZE], addr1: int32, addr2: int32, addr3: int32, lut: float32[1536]):
    """Dispatcher for the vector processing unit."""
    if op == OP_VADD: vadd(mem, addr1, addr2, addr3)
    if op == OP_SCALE: scale(mem, addr1, mem[addr2], addr3)
    if op == OP_RELU: relu(mem, addr1, addr3)
    if op == OP_GELU: gelu(mem, addr1, addr3, lut)
    if op == OP_LN: layer_norm(mem, addr1, addr3, lut)
    if op == OP_EXP: my_exp(mem, addr1, addr3, lut)
    if op == OP_SOFTMAX: softmax(mem, addr1, addr2, addr3, lut)


# ============================================================
# Top-level accelerator
# ============================================================

def tpu(ctrl: int32, d_addr: int32, size: int32,
        data_in: float32[BW], inst_in: int32[BW], outval: float32[BW]):
    """
    Top-level temporal accelerator.

    Holds the shared on-chip memory, the instruction memory, and the
    shared LUT as stateful arrays. Behavior is selected by the `ctrl`
    code issued by the host.
    """
    mem: float32[MEM_SIZE] @ stateful = 0.0
    imem: int32[IMEM_SIZE * 4] @ stateful = 0
    shared_lut: float32[1536] @ stateful = 0.0

    if ctrl == CTRL_MEMCPY_H2D:
        for offset in range(size): mem[d_addr + offset] = data_in[offset]
    if ctrl == CTRL_MEMCPY_D2H:
        for offset in range(size): outval[offset] = mem[d_addr + offset]
    if ctrl == CTRL_MEMCPY_PROG:
        for i in range(size * 4): imem[d_addr * 4 + i] = inst_in[i]
    if ctrl == CTRL_RUN_PROG:
        op: int32 = 0
        pc = d_addr
        while op != OP_END and pc < IMEM_SIZE:
            op = imem[pc * 4]
            addr1: int32 = imem[pc * 4 + 1]
            addr2: int32 = imem[pc * 4 + 2]
            addr3: int32 = imem[pc * 4 + 3]
            if op == OP_MM: matmul(mem, addr1, addr2, addr3, 0)
            if op == OP_MM_ACC: matmul(mem, addr1, addr2, addr3, 1)
            if op == OP_VADD or op == OP_SCALE or op == OP_RELU or op == OP_GELU or op == OP_LN or op == OP_EXP or op == OP_SOFTMAX:
                vpu(op, mem, addr1, addr2, addr3, shared_lut)
            pc += 1
    if ctrl == CTRL_LOAD_LUT:
        for offset in range(1536): shared_lut[offset] = data_in[offset]


# ============================================================
# Schedule
# ============================================================
# Two schedule optimizations are applied on top of the baseline:
#   (1) the matmul k-loop is fully unrolled. The fp32 accumulation has
#       a loop-carried dependency (c = c + a*b) that prevents
#       pipeline II=1, so unrolling is used instead to remove the
#       dependency and shorten latency.
#   (2) all VPU inner loops are pipelined, except exp and gelu, whose
#       LUT-access path lengthens the critical path and causes negative
#       timing slack when pipelined.
# ============================================================

def get_matmul_sch():
    """Build the schedule for the matmul kernel."""
    s = allo.customize(matmul)
    s.unroll("k")
    s.unfold("PE", [0, 1])
    s.partition(s.local_A, dim=0)
    s.partition(s.local_B, dim=1)
    return s


def get_tpu_sch():
    """Build the schedule for the full temporal accelerator."""
    s0 = get_matmul_sch()

    # VPU inner-loop pipelining. exp and gelu are intentionally left
    # un-pipelined; see the schedule note above.
    s1 = allo.customize(vadd)
    s1.pipeline("offset")

    s_exp = allo.customize(my_exp)

    s_gelu = allo.customize(gelu)

    s_ln = allo.customize(layer_norm)
    s_ln.pipeline("i_sum")
    s_ln.pipeline("i_norm")

    s_scale = allo.customize(scale)
    s_scale.pipeline("offset")

    s_relu = allo.customize(relu)
    s_relu.pipeline("offset")

    s_softmax = allo.customize(softmax)
    s_softmax.pipeline("j_max")
    s_softmax.pipeline("j_sub")
    s_softmax.pipeline("j_sum")
    s_softmax.pipeline("j_div")
    s_softmax.compose(s_exp)

    s4 = allo.customize(vpu)
    s4.compose(s1)
    s4.compose(s_exp)
    s4.compose(s_softmax)
    s4.compose(s_scale)
    s4.compose(s_relu)
    s4.compose(s_gelu)
    s4.compose(s_ln)

    s = allo.customize(tpu)
    s.compose(s0)
    s.compose(s4)
    return s


# ============================================================
# Build and run hardware emulation
# ============================================================

if __name__ == "__main__":
    PROJECT_NAME = "Temporal_Accelerator"
    if os.path.exists(PROJECT_NAME):
        shutil.rmtree(PROJECT_NAME)

    s = get_tpu_sch()

    mod = s.build(
        target="vitis_hls",
        mode="hw_emu",
        project=PROJECT_NAME,
        configs={"device": "u280"}
    )

    ctrl_cmd, d_addr, size = np.int32(0), np.int32(0), np.int32(16)
    data_in_array = np.zeros((BW,), dtype=np.float32)
    inst_in_array = np.zeros((BW,), dtype=np.int32)
    outval_array = np.zeros((BW,), dtype=np.float32)

    print(" Running Hardware Emulation (matmul unroll + VPU pipeline w/o exp,gelu)...")
    mod(ctrl_cmd, d_addr, size, data_in_array, inst_in_array, outval_array)

    print("\n Synthesis Finished! Check the csynth.rpt file.")