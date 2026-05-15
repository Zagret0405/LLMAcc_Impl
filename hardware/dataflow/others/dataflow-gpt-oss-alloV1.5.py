from __future__ import annotations
import math
import os
import re
import json
import shutil
import sys
import glob

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

import allo
import numpy as np
import torch
import torch.nn as nn
import allo.dataflow as df
from allo.backend import hls
from allo.ir.types import float32, int32, Stream
from allo.ir.utils import _get_global_vars
import allo.ir.infer
import allo.ir.visitor
import ast
import traceback

MtQ, NtQ = 1 , 1    # max: Mt=L, Nt=D
MtK, NtK = 1 , 1    # max: Mt=L, Nt=Dkv
MtV, NtV = 1 , 1    # max: Mt=L, Nt=Dkv
MtO, NtO = 1 , 1    # max: Mt=L, Nt=D
MtQKT, NtQKT = 1 , 1    # max: Mt=L, Nt=L
MtYV, NtYV = 1 , 1    # max: Mt=L, Nt=Dh
MtMoeGate, NtMoeGate = 1 , 1    # max: Mt=1, Nt=E
MtMoeG, NtMoeG = 1 , 1    # max: Mt=1, Nt=Dffn
MtMoeL, NtMoeL = 1 , 1    # max: Mt=1, Nt=Dffn
MtMoeDown, NtMoeDown = 1 , 1    # max: Mt=1, Nt=D

PQ0, PQ1 = MtQ + 2, NtQ + 2
PK0, PK1 = MtK + 2, NtK + 2
PV0, PV1 = MtV + 2, NtV + 2
PQKT0, PQKT1 = MtQKT + 2, NtQKT + 2
PYV0, PYV1 = MtYV + 2, NtYV + 2
PO0, PO1 = MtO + 2, NtO + 2
PGate0, PGate1 = MtMoeGate + 2, NtMoeGate + 2
PMoeG0, PMoeG1 = MtMoeG + 2, NtMoeG + 2
PMoeL0, PMoeL1 = MtMoeL + 2, NtMoeL + 2
PMoeDown0, PMoeDown1 = MtMoeDown + 2, NtMoeDown + 2



def get_gpt_oss_top(
    Ty, B, HQ, HKV, Qm, L, D, Dkv, Dh, Dh2, Dffn, E, Vocab, Win,
):
    alpha = 1.702
    limit = 7.0

    global MtQ, NtQ, PQ0, PQ1
    global MtK, NtK, PK0, PK1
    global MtV, NtV, PV0, PV1
    global MtO, NtO, PO0, PO1
    global MtQKT, NtQKT, PQKT0, PQKT1
    global MtYV, NtYV, PYV0, PYV1
    global MtMoeGate, NtMoeGate, PGate0, PGate1
    global MtMoeG, NtMoeG, PMoeG0, PMoeG1
    global MtMoeL, NtMoeL, PMoeL0, PMoeL1
    global MtMoeDown, NtMoeDown, PMoeDown0, PMoeDown1
	
    # q_systolic = make_systolic_weight(Ty, B, L, D, D, MtQ, NtQ, "Q")
    # k_systolic = make_systolic_weight(Ty, B, L, D, Dkv, MtK, NtK, "K")
    # v_systolic = make_systolic_weight(Ty, B, L, D, Dkv, MtV, NtV, "V")
    
    # qkt_systolic = make_systolic_bmm(Ty, B * HQ, L, Dh, L, MtQKT, NtQKT, "QKT")
    # yv_systolic = make_systolic_bmm(Ty, B * HQ, L, L, Dh, MtYV, NtYV, "YV")
    
    # o_systolic = make_systolic_weight(Ty, B, L, D, D, MtO, NtO, "O")
    
    # moe_gate_systolic = make_systolic_weight(Ty, B * L, 1, D, E, MtMoeGate, NtMoeGate, "MOE_GATE")
    
    # moe_g_systolic = make_systolic_bmm(float32, B * L * 4, 1, D, Dffn, MtMoeG, NtMoeG, "MOE_G")
    # moe_l_systolic = make_systolic_bmm(float32, B * L * 4, 1, D, Dffn, MtMoeL, NtMoeL, "MOE_L")
    # moe_down_systolic = make_systolic_bmm(float32, B * L * 4, 1, Dffn, D, MtMoeDown, NtMoeDown, "MOE_O")

    @df.region()
    def top(
        X: "float32[B, L, D]", Wq: "float32[D, D]", bq: "float32[D]", Wk: "float32[D, Dkv]", bk: "float32[Dkv]", Wv: "float32[D, Dkv]", bv: "float32[Dkv]", Wo: "float32[D, D]", bo: "float32[D]", Wgate: "float32[D, E]", Wg_experts: "float32[E, D, Dffn]", Wl_experts: "float32[E, D, Dffn]", W2_experts: "float32[E, Dffn, D]", Bg_experts: "float32[E, Dffn]", Bl_experts: "float32[E, Dffn]", B2_experts: "float32[E, D]", norm1_scale: "float32[D]", norm2_scale: "float32[D]", normf_scale: "float32[D]", sinks: "float32[HQ]", cos: "float32[L, Dh2]", sin: "float32[L, Dh2]", Y: "float32[B, L, D]",
    ):
        with allo.meta_if(B > 0 and HQ > 0 and HKV > 0 and Qm > 0 and L > 0 and D > 0 and Dkv > 0 and Dh > 0 and Dh2 > 0 and Dffn > 0 and E > 0 and Vocab > 0 and Win > 0):
            pass
        with allo.meta_if(MtQ > 0 and NtQ > 0 and MtK > 0 and NtK > 0 and MtV > 0 and NtV > 0 and MtO > 0 and NtO > 0 and MtQKT > 0 and NtQKT > 0 and MtYV > 0 and NtYV > 0 and MtMoeGate > 0 and NtMoeGate > 0 and MtMoeG > 0 and NtMoeG > 0 and MtMoeL > 0 and NtMoeL > 0 and MtMoeDown > 0 and NtMoeDown > 0):
            pass
        
        n1: float32[B, L, D]
        Q: float32[B, L, D]
        K: float32[B, L, Dkv]
        V: float32[B, L, Dkv]
        
        Q_rot: float32[B*HQ, L, Dh]
        K_rot: float32[B*HQ, Dh, L]
        V_h: float32[B*HQ, L, Dh]
        
        scores: float32[B*HQ, L, L]
        probs: float32[B*HQ, L, L]
        attn_ctx_h: float32[B*HQ, L, Dh]
        attn_ctx: float32[B, L, D]
        
        O: float32[B, L, D]
        res1: float32[B, L, D]
        n2: float32[B, L, D]
        n2_flat: float32[B * L, 1, D]
        
        gate_logits: float32[B*L, 1, E]
        expert_idx: int32[B*L*4]
        expert_ws: float32[B*L*4]
        
        moe_g_in: float32[B*L*4, 1, D]
        moe_g_w: float32[B*L*4, D, Dffn]
        moe_g_out: float32[B*L*4, 1, Dffn]
        
        moe_l_w: float32[B*L*4, D, Dffn]
        moe_l_out: float32[B*L*4, 1, Dffn]
        
        moe_down_in: float32[B*L*4, 1, Dffn]
        moe_down_w: float32[B*L*4, Dffn, D]
        moe_down_out: float32[B*L*4, 1, D]
        
        moe_reduce_out: float32[B, L, D]

        fifo_A_q: Stream[float32, 4][PQ0,PQ1]
        fifo_B_q: Stream[float32, 4][PQ0,PQ1]
        fifo_A_k: Stream[float32, 4][PK0,PK1]
        fifo_B_k: Stream[float32, 4][PK0,PK1]
        fifo_A_v: Stream[float32, 4][PV0,PV1]
        fifo_B_v: Stream[float32, 4][PV0,PV1]
        fifo_A_qkt: Stream[float32, 4][PQKT0,PQKT1]
        fifo_B_qkt: Stream[float32, 4][PQKT0,PQKT1]
        fifo_A_yv: Stream[float32, 4][PYV0,PYV1]
        fifo_B_yv: Stream[float32, 4][PYV0,PYV1]
        fifo_A_o: Stream[float32, 4][PO0,PO1]
        fifo_B_o: Stream[float32, 4][PO0,PO1]
        fifo_A_gate: Stream[float32, 4][PGate0,PGate1]
        fifo_B_gate: Stream[float32, 4][PGate0,PGate1]
        fifo_A_moe_g: Stream[float32, 4][PMoeG0,PMoeG1]
        fifo_B_moe_g: Stream[float32, 4][PMoeG0,PMoeG1]
        fifo_A_moe_l: Stream[float32, 4][PMoeL0,PMoeL1]
        fifo_B_moe_l: Stream[float32, 4][PMoeL0,PMoeL1]
        fifo_A_moe_down: Stream[float32, 4][PMoeDown0,PMoeDown1]
        fifo_B_moe_down: Stream[float32, 4][PMoeDown0,PMoeDown1]

        @df.kernel(mapping=[1])
        def kernel_norm1():
            for b in range(B):
                mean_sq: float32[L] = 0.0
                for i, j in allo.grid(L, D, name="rms_sum"):
                    mean_sq[i] += X[b, i, j] * X[b, i, j]
                inv_rms: float32[L]
                for i in allo.grid(L, name="rms_inv"):
                    inv_rms[i] = 1.0 / allo.sqrt(mean_sq[i] / float(D) + 0.00001)
                for i, j in allo.grid(L, D, name="rms_apply"):
                    n1[b, i, j] = X[b, i, j] * inv_rms[i] * norm1_scale[j]

        @df.kernel(mapping=[PQ0, PQ1],args=[n1, Wq, Q])
        def q_systolic(
            q_A: float32[B, L, D],
            q_B: float32[D, D],
            q_C: float32[B, L, D],
        ):
            i, j = df.get_pid()
            for b in range(B):
                for m in range(L // MtQ):
                    for n in range(D // NtQ):
                        with allo.meta_if(i in {0, MtQ+1} and j in {0, NtQ+1}):
                            pass
                        with allo.meta_elif(j == 0):
                            for k in range(D):
                                fifo_A_q[i, j+1].put(q_A[b, m*MtQ+i-1, k])
                        with allo.meta_elif(i == 0):
                            for k in range(D):
                                fifo_B_q[i+1, j].put(q_B[k, n*NtQ+j-1])
                        with allo.meta_elif(i == MtQ + 1):
                            for k in range(D):
                                _b: float32 = fifo_B_q[i, j].get()
                        with allo.meta_elif(j == NtQ + 1):
                            for k in range(D):
                                _a: float32 = fifo_A_q[i, j].get()
                        with allo.meta_else():
                            c: float32 = 0.0
                            for k in range(D):
                                a: float32 = fifo_A_q[i, j].get()
                                _b2: float32 = fifo_B_q[i, j].get()
                                c += a * _b2
                                fifo_A_q[i, j+1].put(a)
                                fifo_B_q[i+1, j].put(_b2)
                            q_C[b, m*MtQ+i-1, n*NtQ+j-1] = c

        @df.kernel(mapping=[PK0, PK1],args=[n1, Wk, K])
        def k_systolic(
            k_A: float32[B, L, D],
            k_B: float32[D, Dkv],
            k_C: float32[B, L, Dkv],
        ):
            i, j = df.get_pid()
            for b in range(B):
                for m in range(L // MtK):
                    for n in range(Dkv // NtK):
                        with allo.meta_if(i in {0, MtK+1} and j in {0, NtK+1}):
                            pass
                        with allo.meta_elif(j == 0):
                            for k in range(D):
                                fifo_A_k[i, j+1].put(k_A[b, m*MtK+i-1, k])
                        with allo.meta_elif(i == 0):
                            for k in range(D):
                                fifo_B_k[i+1, j].put(k_B[k, n*NtK+j-1])
                        with allo.meta_elif(i == MtK + 1):
                            for k in range(D):
                                _b: float32 = fifo_B_k[i, j].get()
                        with allo.meta_elif(j == NtK + 1):
                            for k in range(D):
                                _a: float32 = fifo_A_k[i, j].get()
                        with allo.meta_else():
                            c: float32 = 0.0
                            for k in range(D):
                                a: float32 = fifo_A_k[i, j].get()
                                _b2: float32 = fifo_B_k[i, j].get()
                                c += a * _b2
                                fifo_A_k[i, j+1].put(a)
                                fifo_B_k[i+1, j].put(_b2)
                            k_C[b, m*MtK+i-1, n*NtK+j-1] = c

        @df.kernel(mapping=[PV0, PV1],args=[n1, Wv, V])
        def v_systolic(
            v_A: float32[B, L, D],
            v_B: float32[D, Dkv],
            v_C: float32[B, L, Dkv],
        ):
            i, j = df.get_pid()
            for b in range(B):
                for m in range(L // MtV):
                    for n in range(Dkv // NtV):
                        with allo.meta_if(i in {0, MtV+1} and j in {0, NtV+1}):
                            pass
                        with allo.meta_elif(j == 0):
                            for k in range(D):
                                fifo_A_v[i, j+1].put(v_A[b, m*MtV+i-1, k])
                        with allo.meta_elif(i == 0):
                            for k in range(D):
                                fifo_B_v[i+1, j].put(v_B[k, n*NtV+j-1])
                        with allo.meta_elif(i == MtV + 1):
                            for k in range(D):
                                _b: float32 = fifo_B_v[i, j].get()
                        with allo.meta_elif(j == NtV + 1):
                            for k in range(D):
                                _a: float32 = fifo_A_v[i, j].get()
                        with allo.meta_else():
                            c: float32 = 0.0
                            for k in range(D):
                                a: float32 = fifo_A_v[i, j].get()
                                _b2: float32 = fifo_B_v[i, j].get()
                                c += a * _b2
                                fifo_A_v[i, j+1].put(a)
                                fifo_B_v[i+1, j].put(_b2)
                            v_C[b, m*MtV+i-1, n*NtV+j-1] = c   
                                 
        #q_systolic(n1, Wq, Q, fifo_A_q, fifo_B_q)
        #k_systolic(n1, Wk, K, fifo_A_k, fifo_B_k)
        #v_systolic(n1, Wv, V, fifo_A_v, fifo_B_v)

        @df.kernel(mapping=[1])
        def kernel_mha_prep():
            for b in range(B):
                for i, j in allo.grid(L, D, name="bias_q"):
                    Q[b, i, j] = Q[b, i, j] + bq[j]
                for i, j in allo.grid(L, Dkv, name="bias_k"):
                    K[b, i, j] = K[b, i, j] + bk[j]
                for i, j in allo.grid(L, Dkv, name="bias_v"):
                    V[b, i, j] = V[b, i, j] + bv[j]
                for kv in range(HKV):
                    for i, j in allo.grid(L, Dh2, name="k_rope"):
                        x1: float32 = K[b, i, kv * Dh + j]
                        x2: float32 = K[b, i, kv * Dh + j + Dh2]
                        c: float32 = cos[i, j]
                        s: float32 = sin[i, j]
                        r1: float32 = x1 * c - x2 * s
                        r2: float32 = x2 * c + x1 * s
                        for qm in range(Qm):
                            h = kv * Qm + qm
                            bh = b * HQ + h
                            K_rot[bh, j, i] = r1
                            K_rot[bh, j + Dh2, i] = r2
                            V_h[bh, i, j] = V[b, i, kv * Dh + j]
                            V_h[bh, i, j + Dh2] = V[b, i, kv * Dh + j + Dh2]
                    if Dh2 * 2 < Dh:
                        for i in allo.grid(L):
                            for qm in range(Qm):
                                h = kv * Qm + qm
                                bh = b * HQ + h
                                K_rot[bh, Dh - 1, i] = K[b, i, kv * Dh + Dh - 1]
                                V_h[bh, i, Dh - 1] = V[b, i, kv * Dh + Dh - 1]
                    for qm in range(Qm):
                        h = kv * Qm + qm
                        bh = b * HQ + h
                        for i, j in allo.grid(L, Dh2, name="q_rope"):
                            x1_q: float32 = Q[b, i, h * Dh + j]
                            x2_q: float32 = Q[b, i, h * Dh + j + Dh2]
                            c_q: float32 = cos[i, j]
                            s_q: float32 = sin[i, j]
                            Q_rot[bh, i, j] = x1_q * c_q - x2_q * s_q
                            Q_rot[bh, i, j + Dh2] = x2_q * c_q + x1_q * s_q
                        if Dh2 * 2 < Dh:
                            for i in allo.grid(L):
                                Q_rot[bh, i, Dh - 1] = Q[b, i, h * Dh + Dh - 1]

        @df.kernel(mapping=[PQKT0, PQKT1],args=[Q_rot, K_rot, scores])
        def qkt_systolic(
            qkt_A: float32[B * HQ, L, Dh],
            qkt_B: float32[B * HQ, Dh, L],
            qkt_C: float32[B * HQ, L, L],
        ):
            i, j = df.get_pid()
            for b in range(B * HQ):
                for m in range(L // MtQKT):
                    for n in range(L // NtQKT):
                        with allo.meta_if(i in {0, MtQKT+1} and j in {0, NtQKT+1}):
                            pass
                        with allo.meta_elif(j == 0):
                            for k in range(Dh):
                                fifo_A_qkt[i, j+1].put(qkt_A[b, m*MtQKT+i-1, k])
                        with allo.meta_elif(i == 0):
                            for k in range(Dh):
                                fifo_B_qkt[i+1, j].put(qkt_B[b, k, n*NtQKT+j-1])
                        with allo.meta_elif(i == MtQKT + 1):
                            for k in range(Dh):
                                _b: float32 = fifo_B_qkt[i, j].get()
                        with allo.meta_elif(j == NtQKT + 1):
                            for k in range(Dh):
                                _a: float32 = fifo_A_qkt[i, j].get()
                        with allo.meta_else():
                            c: float32 = 0.0
                            for k in range(Dh):
                                a: float32 = fifo_A_qkt[i, j].get()
                                _b2: float32 = fifo_B_qkt[i, j].get()
                                c += a * _b2
                                fifo_A_qkt[i, j+1].put(a)
                                fifo_B_qkt[i+1, j].put(_b2)
                            qkt_C[b, m*MtQKT+i-1, n*NtQKT+j-1] = c

        #qkt_systolic(Q_rot, K_rot, scores, fifo_A_qkt, fifo_B_qkt)

        @df.kernel(mapping=[1])
        def kernel_softmax():
            for bh in range(B * HQ):
                h = bh % HQ
                E_buf: float32[L, L]
                M: float32[L] = -1000000000.0
                S: float32[L] = 0.0
                for i, j in allo.grid(L, L, name="smx_max"):
                    v: float32 = scores[bh, i, j] / allo.sqrt(float(Dh))
                    if j > i: v = -1000000000.0
                    if Win > 0 and j + Win <= i: v = -1000000000.0
                    if v > M[i]: M[i] = v
                for i in allo.grid(L, name="smx_sink_max"):
                    if sinks[h] > M[i]: M[i] = sinks[h]
                for i, j in allo.grid(L, L, name="smx_exp"):
                    v: float32 = scores[bh, i, j] / allo.sqrt(float(Dh))
                    if j > i: v = -1000000000.0
                    if Win > 0 and j + Win <= i: v = -1000000000.0
                    E_buf[i, j] = allo.exp(v - M[i])
                    S[i] += E_buf[i, j]
                for i in allo.grid(L, name="smx_sink_sum"):
                    S[i] += allo.exp(sinks[h] - M[i])
                for i, j in allo.grid(L, L, name="smx_update"):
                    probs[bh, i, j] = E_buf[i, j] / S[i]

        @df.kernel(mapping=[PYV0, PYV1],args=[probs, V_h, attn_ctx_h])
        def yv_systolic(
            yv_A: float32[B * HQ, L, L],
            yv_B: float32[B * HQ, L, Dh],
            yv_C: float32[B * HQ, L, Dh],
        ):
            i, j = df.get_pid()
            for b in range(B * HQ):
                for m in range(L // MtYV):
                    for n in range(Dh // NtYV):
                        with allo.meta_if(i in {0, MtYV+1} and j in {0, NtYV+1}):
                            pass
                        with allo.meta_elif(j == 0):
                            for k in range(L):
                                fifo_A_yv[i, j+1].put(yv_A[b, m*MtYV+i-1, k])
                        with allo.meta_elif(i == 0):
                            for k in range(L):
                                fifo_B_yv[i+1, j].put(yv_B[b, k, n*NtYV+j-1])
                        with allo.meta_elif(i == MtYV + 1):
                            for k in range(L):
                                _b: float32 = fifo_B_yv[i, j].get()
                        with allo.meta_elif(j == NtYV + 1):
                            for k in range(L):
                                _a: float32 = fifo_A_yv[i, j].get()
                        with allo.meta_else():
                            c: float32 = 0.0
                            for k in range(L):
                                a: float32 = fifo_A_yv[i, j].get()
                                _b2: float32 = fifo_B_yv[i, j].get()
                                c += a * _b2
                                fifo_A_yv[i, j+1].put(a)
                                fifo_B_yv[i+1, j].put(_b2)
                            yv_C[b, m*MtYV+i-1, n*NtYV+j-1] = c
        #yv_systolic(probs, V_h, attn_ctx_h, fifo_A_yv, fifo_B_yv)

        @df.kernel(mapping=[1])
        def kernel_mha_merge():
            for b in range(B):
                for h in range(HQ):
                    bh = b * HQ + h
                    for i, j in allo.grid(L, Dh, name="merge"):
                        attn_ctx[b, i, h * Dh + j] = attn_ctx_h[bh, i, j]

        @df.kernel(mapping=[PO0, PO1],args=[attn_ctx, Wo, O])
        def o_systolic(
            o_A: float32[B, L, D],
            o_B: float32[D, D],
            o_C: float32[B, L, D],
        ):
            i, j = df.get_pid()
            for b in range(B):
                for m in range(L // MtO):
                    for n in range(D // NtO):
                        with allo.meta_if(i in {0, MtO+1} and j in {0, NtO+1}):
                            pass
                        with allo.meta_elif(j == 0):
                            for k in range(D):
                                fifo_A_o[i, j+1].put(o_A[b, m*MtO+i-1, k])
                        with allo.meta_elif(i == 0):
                            for k in range(D):
                                fifo_B_o[i+1, j].put(o_B[k, n*NtO+j-1])
                        with allo.meta_elif(i == MtO + 1):
                            for k in range(D):
                                _b: float32 = fifo_B_o[i, j].get()
                        with allo.meta_elif(j == NtO + 1):
                            for k in range(D):
                                _a: float32 = fifo_A_o[i, j].get()
                        with allo.meta_else():
                            c: float32 = 0.0
                            for k in range(D):
                                a: float32 = fifo_A_o[i, j].get()
                                _b2: float32 = fifo_B_o[i, j].get()
                                c += a * _b2
                                fifo_A_o[i, j+1].put(a)
                                fifo_B_o[i+1, j].put(_b2)
                            o_C[b, m*MtO+i-1, n*NtO+j-1] = c
        # o_systolic(attn_ctx, Wo, O, fifo_A_o, fifo_B_o)

        @df.kernel(mapping=[1])
        def kernel_add_norm2_and_prep_moe():
            for b in range(B):
                mean_sq2: float32[L] = 0.0
                for i, j in allo.grid(L, D, name="add_res1"):
                    res1[b, i, j] = X[b, i, j] + O[b, i, j] + bo[j]
                    mean_sq2[i] += res1[b, i, j] * res1[b, i, j]
                inv_rms2: float32[L]
                for i in allo.grid(L):
                    inv_rms2[i] = 1.0 / allo.sqrt(mean_sq2[i] / float(D) + 0.00001)
                for i, j in allo.grid(L, D, name="norm2"):
                    v: float32 = res1[b, i, j] * inv_rms2[i] * norm2_scale[j]
                    n2[b, i, j] = v
                    n2_flat[b * L + i, 0, j] = v
                    # Flatten n2 for batched MOE systolic array
                    moe_g_in[(b * L + i) * 4 + 0, 0, j] = v
                    moe_g_in[(b * L + i) * 4 + 1, 0, j] = v
                    moe_g_in[(b * L + i) * 4 + 2, 0, j] = v
                    moe_g_in[(b * L + i) * 4 + 3, 0, j] = v

        @df.kernel(mapping=[PGate0, PGate1],args=[n2, Wgate, gate_logits])
        def moe_gate_systolic(
            gate_A: float32[B, L, D],
            gate_B: float32[D, E],
            gate_C: float32[B * L, 1, E],
        ):
            i, j = df.get_pid()
            for b in range(B):
                for l in range(L):
                    bl = b * L + l
                    for m in range(1 // MtMoeGate):
                        for n in range(E // NtMoeGate):
                            with allo.meta_if(i in {0, MtMoeGate+1} and j in {0, NtMoeGate+1}):
                                pass
                            with allo.meta_elif(j == 0):
                                for k in range(D):
                                    fifo_A_gate[i, j+1].put(gate_A[b, l, k])
                            with allo.meta_elif(i == 0):
                                for k in range(D):
                                    fifo_B_gate[i+1, j].put(gate_B[k, n*NtMoeGate+j-1])
                            with allo.meta_elif(i == MtMoeGate + 1):
                                for k in range(D):
                                    _b: float32 = fifo_B_gate[i, j].get()
                            with allo.meta_elif(j == NtMoeGate + 1):
                                for k in range(D):
                                    _a: float32 = fifo_A_gate[i, j].get()
                            with allo.meta_else():
                                c: float32 = 0.0
                                for k in range(D):
                                    a: float32 = fifo_A_gate[i, j].get()
                                    _b2: float32 = fifo_B_gate[i, j].get()
                                    c += a * _b2
                                    fifo_A_gate[i, j+1].put(a)
                                    fifo_B_gate[i+1, j].put(_b2)
                                gate_C[bl, m*MtMoeGate+i-1, n*NtMoeGate+j-1] = c
        #moe_gate_systolic(n2, Wgate, gate_logits, fifo_A_gate, fifo_B_gate)

        @df.kernel(mapping=[1])
        def kernel_moe_topk_dispatch():
            neg_inf: float32 = -1000000000.0
            for b in range(B):
                for i in range(L):
                    bl = b * L + i
                    i0 = 0
                    i1 = 0
                    i2 = 0
                    i3 = 0
                    v0: float32 = neg_inf
                    v1: float32 = neg_inf
                    v2: float32 = neg_inf
                    v3: float32 = neg_inf
                    for e in range(E):
                        v: float32 = gate_logits[bl, 0, e]
                        if v > v0:
                            v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v0; i1 = i0; v0 = v; i0 = e
                        elif v > v1:
                            v3 = v2; i3 = i2; v2 = v1; i2 = i1; v1 = v; i1 = e
                        elif v > v2:
                            v3 = v2; i3 = i2; v2 = v; i2 = e
                        elif v > v3:
                            v3 = v; i3 = e
                    m: float32 = v0
                    w0: float32 = allo.exp(v0 - m)
                    w1: float32 = allo.exp(v1 - m)
                    w2: float32 = allo.exp(v2 - m)
                    w3: float32 = allo.exp(v3 - m)
                    sumw: float32 = w0 + w1 + w2 + w3
                    
                    for s in range(4):
                        e = 0
                        ws: float32 = 0.0
                        if s == 0: e = i0; ws = w0 / sumw
                        elif s == 1: e = i1; ws = w1 / sumw
                        elif s == 2: e = i2; ws = w2 / sumw
                        else: e = i3; ws = w3 / sumw
                        
                        idx = bl * 4 + s
                        if s == 0:
                            expert_idx[idx] = i0
                        elif s == 1:
                            expert_idx[idx] = i1
                        elif s == 2:
                            expert_idx[idx] = i2
                        else:
                            expert_idx[idx] = i3
                        expert_ws[idx] = ws
                        for d, f in allo.grid(D, Dffn):
                            moe_g_w[idx, d, f] = Wg_experts[e, d, f]
                            moe_l_w[idx, d, f] = Wl_experts[e, d, f]

        @df.kernel(mapping=[PMoeG0, PMoeG1],args=[moe_g_in, moe_g_w, moe_g_out])
        def moe_g_systolic(
            moe_g_A: float32[B * L * 4, 1, D],
            moe_g_B: float32[B * L * 4, D, Dffn],
            moe_g_C: float32[B * L * 4, 1, Dffn],
        ):
            i, j = df.get_pid()
            for b in range(B * L * 4):
                for m in range(1 // MtMoeG):
                    for n in range(Dffn // NtMoeG):
                        with allo.meta_if(i in {0, MtMoeG+1} and j in {0, NtMoeG+1}):
                            pass
                        with allo.meta_elif(j == 0):
                            for k in range(D):
                                fifo_A_moe_g[i, j+1].put(moe_g_A[b, m*MtMoeG+i-1, k])
                        with allo.meta_elif(i == 0):
                            for k in range(D):
                                fifo_B_moe_g[i+1, j].put(moe_g_B[b, k, n*NtMoeG+j-1])
                        with allo.meta_elif(i == MtMoeG + 1):
                            for k in range(D):
                                _b: float32 = fifo_B_moe_g[i, j].get()
                        with allo.meta_elif(j == NtMoeG + 1):
                            for k in range(D):
                                _a: float32 = fifo_A_moe_g[i, j].get()
                        with allo.meta_else():
                            c: float32 = 0.0
                            for k in range(D):
                                a: float32 = fifo_A_moe_g[i, j].get()
                                _b2: float32 = fifo_B_moe_g[i, j].get()
                                c += a * _b2
                                fifo_A_moe_g[i, j+1].put(a)
                                fifo_B_moe_g[i+1, j].put(_b2)
                            moe_g_C[b, m*MtMoeG+i-1, n*NtMoeG+j-1] = c

        @df.kernel(mapping=[PMoeL0,PMoeL1],args=[moe_g_in, moe_l_w, moe_l_out])
        def moe_l_systolic(
            moe_l_A: float32[B * L * 4, 1, D],
            moe_l_B: float32[B * L * 4, D, Dffn],
            moe_l_C: float32[B * L * 4, 1, Dffn],
        ):
            i, j = df.get_pid()
            for b in range(B * L * 4):
                for m in range(1 // MtMoeL):
                    for n in range(Dffn // NtMoeL):
                        with allo.meta_if(i in {0, MtMoeL+1} and j in {0, NtMoeL+1}):
                            pass
                        with allo.meta_elif(j == 0):
                            for k in range(D):
                                fifo_A_moe_l[i, j+1].put(moe_l_A[b, m*MtMoeL+i-1, k])
                        with allo.meta_elif(i == 0):
                            for k in range(D):
                                fifo_B_moe_l[i+1, j].put(moe_l_B[b, k, n*NtMoeL+j-1])
                        with allo.meta_elif(i == MtMoeL + 1):
                            for k in range(D):
                                _b: float32 = fifo_B_moe_l[i, j].get()
                        with allo.meta_elif(j == NtMoeL + 1):
                            for k in range(D):
                                _a: float32 = fifo_A_moe_l[i, j].get()
                        with allo.meta_else():
                            c: float32 = 0.0
                            for k in range(D):
                                a: float32 = fifo_A_moe_l[i, j].get()
                                _b2: float32 = fifo_B_moe_l[i, j].get()
                                c += a * _b2
                                fifo_A_moe_l[i, j+1].put(a)
                                fifo_B_moe_l[i+1, j].put(_b2)
                            moe_l_C[b, m*MtMoeL+i-1, n*NtMoeL+j-1] = c
        # moe_g_systolic(moe_g_in, moe_g_w, moe_g_out, fifo_A_moe_g, fifo_B_moe_g)
        # moe_l_systolic(moe_g_in, moe_l_w, moe_l_out, fifo_A_moe_l, fifo_B_moe_l)

        @df.kernel(mapping=[1])
        def kernel_moe_swiglu_and_down_dispatch():
            for idx in range(B * L * 4):
                e = expert_idx[idx]
                for f in range(Dffn):
                    xg: float32 = moe_g_out[idx, 0, f] + Bg_experts[e, f]
                    xl: float32 = moe_l_out[idx, 0, f] + Bl_experts[e, f]
                    if xg > limit: xg = limit
                    if xl > limit: xl = limit
                    if xl < -limit: xl = -limit
                    sig: float32 = 1.0 / (1.0 + allo.exp(-alpha * xg))
                    moe_down_in[idx, 0, f] = (xg * sig) * (xl + 1.0)
                ws: float32 = expert_ws[idx]
                for f, d in allo.grid(Dffn, D):
                    moe_down_w[idx, f, d] = W2_experts[e, f, d] * ws

        @df.kernel(mapping=[PMoeDown0, PMoeDown1],args=[moe_down_in, moe_down_w, moe_down_out])
        def moe_down_systolic(
            moe_down_A: float32[B * L * 4, 1, Dffn],
            moe_down_B: float32[B * L * 4, Dffn, D],
            moe_down_C: float32[B * L * 4, 1, D],
        ):
            i, j = df.get_pid()
            for b in range(B * L * 4):
                for m in range(1 // MtMoeDown):
                    for n in range(D // NtMoeDown):
                        with allo.meta_if(i in {0, MtMoeDown+1} and j in {0, NtMoeDown+1}):
                            pass
                        with allo.meta_elif(j == 0):
                            for k in range(Dffn):
                                fifo_A_moe_down[i, j+1].put(moe_down_A[b, m*MtMoeDown+i-1, k])
                        with allo.meta_elif(i == 0):
                            for k in range(Dffn):
                                fifo_B_moe_down[i+1, j].put(moe_down_B[b, k, n*NtMoeDown+j-1])
                        with allo.meta_elif(i == MtMoeDown + 1):
                            for k in range(Dffn):
                                _b: float32 = fifo_B_moe_down[i, j].get()
                        with allo.meta_elif(j == NtMoeDown + 1):
                            for k in range(Dffn):
                                _a: float32 = fifo_A_moe_down[i, j].get()
                        with allo.meta_else():
                            c: float32 = 0.0
                            for k in range(Dffn):
                                a: float32 = fifo_A_moe_down[i, j].get()
                                _b2: float32 = fifo_B_moe_down[i, j].get()
                                c += a * _b2
                                fifo_A_moe_down[i, j+1].put(a)
                                fifo_B_moe_down[i+1, j].put(_b2)
                            moe_down_C[b, m*MtMoeDown+i-1, n*NtMoeDown+j-1] = c
        #moe_down_systolic(moe_down_in, moe_down_w, moe_down_out, fifo_A_moe_down, fifo_B_moe_down)

        @df.kernel(mapping=[1])
        def kernel_moe_reduce_and_final_norm():
            for b in range(B):
                mean_sq_f: float32[L] = 0.0
                for i in range(L):
                    bl = b * L + i
                    for d in range(D):
                        s0 = bl * 4 + 0
                        s1 = bl * 4 + 1
                        s2 = bl * 4 + 2
                        s3 = bl * 4 + 3
                        e0 = expert_idx[s0]
                        e1 = expert_idx[s1]
                        e2 = expert_idx[s2]
                        e3 = expert_idx[s3]
                        w0 = expert_ws[s0]
                        w1 = expert_ws[s1]
                        w2 = expert_ws[s2]
                        w3 = expert_ws[s3]
                        
                        out_val: float32 = 0.0
                        out_val += (moe_down_out[s0, 0, d] + B2_experts[e0, d] * w0)
                        out_val += (moe_down_out[s1, 0, d] + B2_experts[e1, d] * w1)
                        out_val += (moe_down_out[s2, 0, d] + B2_experts[e2, d] * w2)
                        out_val += (moe_down_out[s3, 0, d] + B2_experts[e3, d] * w3)
                        
                        moe_reduce_out[b, i, d] = out_val
                        out: float32 = res1[b, i, d] + out_val
                        mean_sq_f[i] += out * out
                inv_rms_f: float32[L]
                for i in allo.grid(L):
                    inv_rms_f[i] = 1.0 / allo.sqrt(mean_sq_f[i] / float(D) + 0.00001)
                for i, j in allo.grid(L, D):
                    out: float32 = res1[b, i, j] + moe_reduce_out[b, i, j]
                    Y[b, i, j] = out * inv_rms_f[i] * normf_scale[j]

    s = df.customize(top)
    return s
# def build_systolic_tiles(default_mt, default_nt):
# 	# Centralized systolic tile config. Change per-id Mt/Nt here.
	
# 	# tiles = {
# 	# 	"Q": (4, 4),  # max: Mt=L, Nt=D
# 	# 	"K": (4, 4),  # max: Mt=L, Nt=Dkv
# 	# 	"V": (4, 4),  # max: Mt=L, Nt=Dkv
# 	# 	"O": (4, 4),  # max: Mt=L, Nt=D
# 	# 	"QKT": (8, 4),  # max: Mt=L, Nt=L
# 	# 	"YV": (8, 4),  # max: Mt=L, Nt=Dh
# 	# 	"MOE_GATE": (1, 4),  # max: Mt=1, Nt=E
# 	# 	"MOE_G": (1, 64),  # max: Mt=1, Nt=Dffn
# 	# 	"MOE_L": (1, 64),  # max: Mt=1, Nt=Dffn
# 	# 	"MOE_O": (1, 64),  # max: Mt=1, Nt=D
# 	# }
# 	tiles = {
# 		"Q": (1, 1),         # max: Mt=L, Nt=D
# 		"K": (1, 1),         # max: Mt=L, Nt=Dkv
# 		"V": (1, 1),         # max: Mt=L, Nt=Dkv
# 		"O": (1, 1),         # max: Mt=L, Nt=D
# 		"QKT": (1, 1),       # max: Mt=L, Nt=L
# 		"YV": (1, 1),        # max: Mt=L, Nt=Dh
# 		"MOE_GATE": (1, 1),  # max: Mt=1, Nt=E
# 		"MOE_G": (1, 1),     # max: Mt=1, Nt=Dffn
# 		"MOE_L": (1, 1),     # max: Mt=1, Nt=Dffn
# 		"MOE_O": (1, 1),     # max: Mt=1, Nt=D
# 	}

# 	# Make command-line mt/nt effective for attention systolic arrays.
# 	# MOE arrays keep their hand-tuned defaults because Mt must stay 1.
# 	if default_mt is not None and default_nt is not None and (int(default_mt), int(default_nt)) != (1, 1):
# 		mt = max(1, int(default_mt))
# 		nt = max(1, int(default_nt))
# 		for key in ("Q", "K", "V", "O", "QKT", "YV"):
# 			tiles[key] = (mt, nt)

# 	return tiles


def _parse_latency_cycles_from_csynth(report_path):
	"""Extract top-level latency cycles from a Vitis csynth report."""
	latency_re = re.compile(r"\|\s*(\d+)\|\s*(\d+)\|\s*([0-9.]+\s*\w+)\|\s*([0-9.]+\s*\w+)\|\s*(\d+)\|\s*(\d+)\|")
	with open(report_path, "r", encoding="utf-8") as f:
		for line in f:
			m = latency_re.search(line)
			if m:
				return int(m.group(2))
	return None


def _find_csynth_report(project_dir):
	pattern = os.path.join(project_dir, "**", "*csynth.rpt")
	matches = glob.glob(pattern, recursive=True)
	return matches[0] if matches else None


def estimate_theoretical_macs(L, D, Dkv, Dh, HQ, E, Dffn, topk):
	"""Estimate MAC count for one gpt_oss_block execution.

	This follows the major compute structure in the generated kernel:
	Q, K, V, QK^T, Attn*V, O, MOE gate, and top-k expert MLPs.
	"""
	q_macs = L * D * D
	k_macs = L * D * Dkv
	v_macs = L * D * Dkv
	attn_qk_macs = HQ * L * Dh * L
	attn_v_macs = HQ * L * L * Dh
	o_macs = L * D * D
	moe_gate_macs = L * D * E
	moe_up_macs = L * topk * D * Dffn
	moe_linear_macs = L * topk * D * Dffn
	moe_down_macs = L * topk * Dffn * D

	return (
		q_macs
		+ k_macs
		+ v_macs
		+ attn_qk_macs
		+ attn_v_macs
		+ o_macs
		+ moe_gate_macs
		+ moe_up_macs
		+ moe_linear_macs
		+ moe_down_macs
	)


def build_gpt_oss_schedule(B, HQ, HKV, Qm, L, D, Dkv, Dh, Dh2, Dffn, E, Vocab, Win):
     
    global MtQ, NtQ, PQ0, PQ1
    global MtK, NtK, PK0, PK1
    global MtV, NtV, PV0, PV1
    global MtO, NtO, PO0, PO1
    global MtQKT, NtQKT, PQKT0, PQKT1
    global MtYV, NtYV, PYV0, PYV1
    global MtMoeGate, NtMoeGate, PGate0, PGate1
    global MtMoeG, NtMoeG, PMoeG0, PMoeG1
    global MtMoeL, NtMoeL, PMoeL0, PMoeL1
    global MtMoeDown, NtMoeDown, PMoeDown0, PMoeDown1

    top = get_gpt_oss_top(
		float32,
		B,
		HQ,
		HKV,
		Qm,
		L,
		D,
		Dkv,
		Dh,
		Dh2,
		Dffn,
		E,
		Vocab,
		Win,
	)
	
    return top


class RMSNorm(nn.Module):
	def __init__(self, d, eps=1e-5):
		super().__init__()
		self.eps = eps
		self.scale = nn.Parameter(torch.ones(d))

	def forward(self, x):
		ms = torch.mean(x * x, dim=-1, keepdim=True)
		return x * torch.rsqrt(ms + self.eps) * self.scale


class TinyGptOssRef(nn.Module):
	def __init__(self, d, hq, hkv, dffn, vocab, sliding_window, num_experts, experts_per_token):
		super().__init__()
		self.hq = hq
		self.hkv = hkv
		self.q_mult = hq // hkv
		self.dh = d // hq
		self.sliding_window = sliding_window
		self.dkv = hkv * self.dh
		self.num_experts = num_experts
		self.experts_per_token = experts_per_token
		self.norm1 = RMSNorm(d)
		self.norm2 = RMSNorm(d)
		self.normf = RMSNorm(d)
		self.wq = nn.Linear(d, d)
		self.wk = nn.Linear(d, self.dkv)
		self.wv = nn.Linear(d, self.dkv)
		self.wo = nn.Linear(d, d)
		self.wgate = nn.Linear(d, num_experts, bias=False)
		self.wg_experts = nn.Parameter(torch.empty(num_experts, d, dffn))
		self.wl_experts = nn.Parameter(torch.empty(num_experts, d, dffn))
		self.w2_experts = nn.Parameter(torch.empty(num_experts, dffn, d))
		self.bg_experts = nn.Parameter(torch.empty(num_experts, dffn))
		self.bl_experts = nn.Parameter(torch.empty(num_experts, dffn))
		self.b2_experts = nn.Parameter(torch.empty(num_experts, d))
		self.sinks = nn.Parameter(torch.zeros(hq))

		# Initialize expert weights around dense FFN weights for stable parity checks.
		with torch.no_grad():
			for e in range(num_experts):
				nn.init.normal_(self.wg_experts[e], mean=0.0, std=0.02)
				nn.init.normal_(self.wl_experts[e], mean=0.0, std=0.02)
				nn.init.normal_(self.w2_experts[e], mean=0.0, std=0.02)
				nn.init.normal_(self.bg_experts[e], mean=0.0, std=0.02)
				nn.init.normal_(self.bl_experts[e], mean=0.0, std=0.02)
				nn.init.normal_(self.b2_experts[e], mean=0.0, std=0.02)

	def _moe_forward(self, n2):
		logits = torch.einsum("bld,de->ble", n2, self.wgate.weight.T)
		topk = min(4, self.experts_per_token, self.num_experts)
		experts = torch.topk(logits, k=topk, dim=-1, sorted=True)
		weights = torch.softmax(experts.values, dim=-1)
		idx = experts.indices
		if topk < 4:
			pad_w = torch.zeros(n2.shape[0], n2.shape[1], 4 - topk, device=n2.device, dtype=n2.dtype)
			pad_i = torch.zeros(n2.shape[0], n2.shape[1], 4 - topk, device=n2.device, dtype=idx.dtype)
			weights = torch.cat([weights, pad_w], dim=-1)
			idx = torch.cat([idx, pad_i], dim=-1)

		wg_sel = self.wg_experts[idx]  # [B, L, 4, D, Dffn]
		wl_sel = self.wl_experts[idx]  # [B, L, 4, D, Dffn]
		w2_sel = self.w2_experts[idx]  # [B, L, 4, Dffn, D]
		bg_sel = self.bg_experts[idx]  # [B, L, 4, Dffn]
		bl_sel = self.bl_experts[idx]  # [B, L, 4, Dffn]
		b2_sel = self.b2_experts[idx]  # [B, L, 4, D]

		xg = torch.einsum("blkdf,bld->blkf", wg_sel, n2) + bg_sel
		xl = torch.einsum("blkdf,bld->blkf", wl_sel, n2) + bl_sel
		xg = torch.clamp(xg, max=7.0)
		xl = torch.clamp(xl, min=-7.0, max=7.0)
		hidden = (xg * torch.sigmoid(1.702 * xg)) * (xl + 1.0)
		expert_out = torch.einsum("blkfd,blkf->blkd", w2_sel, hidden) + b2_sel
		return torch.sum(expert_out * weights.unsqueeze(-1), dim=2)

	def forward(self, x, cos, sin):
		n1 = self.norm1(x)
		q = self.wq(n1)
		k = self.wk(n1)
		v = self.wv(n1)

		q = q.view(q.shape[0], q.shape[1], self.hq, self.dh)
		k = k.view(k.shape[0], k.shape[1], self.hkv, self.dh)
		v = v.view(v.shape[0], v.shape[1], self.hkv, self.dh)

		dh2 = self.dh // 2
		q1 = q[..., :dh2]
		q2 = q[..., dh2 : 2 * dh2]
		k1 = k[..., :dh2]
		k2 = k[..., dh2 : 2 * dh2]
		cos_b = cos.unsqueeze(0).unsqueeze(2)
		sin_b = sin.unsqueeze(0).unsqueeze(2)
		q_rot = torch.cat([q1 * cos_b - q2 * sin_b, q2 * cos_b + q1 * sin_b], dim=-1)
		k_rot = torch.cat([k1 * cos_b - k2 * sin_b, k2 * cos_b + k1 * sin_b], dim=-1)
		if 2 * dh2 < self.dh:
			q = torch.cat([q_rot, q[..., 2 * dh2 :]], dim=-1)
			k = torch.cat([k_rot, k[..., 2 * dh2 :]], dim=-1)
		else:
			q = q_rot
			k = k_rot

		k = (
			k.unsqueeze(3)
			.expand(k.shape[0], k.shape[1], self.hkv, self.q_mult, self.dh)
			.reshape(k.shape[0], k.shape[1], self.hq, self.dh)
		)
		v = (
			v.unsqueeze(3)
			.expand(v.shape[0], v.shape[1], self.hkv, self.q_mult, self.dh)
			.reshape(v.shape[0], v.shape[1], self.hq, self.dh)
		)

		q = q.permute(0, 2, 1, 3)
		k = k.permute(0, 2, 1, 3)
		v = v.permute(0, 2, 1, 3)

		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
		L = scores.shape[-1]
		causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1)
		scores = scores + causal_mask * -1e10
		if self.sliding_window > 0:
			window_mask = torch.tril(torch.ones(L, L, device=x.device), diagonal=-self.sliding_window)
			scores = scores + window_mask * -1e10

		sink_col = self.sinks.view(1, self.hq, 1, 1).expand(scores.shape[0], self.hq, L, 1)
		scores_plus_sink = torch.cat([scores, sink_col], dim=-1)
		probs = torch.softmax(scores_plus_sink, dim=-1)[..., :L]
		ctx = torch.matmul(probs, v)
		ctx = ctx.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)

		o = self.wo(ctx)
		res1 = x + o

		n2 = self.norm2(res1)
		out = res1 + self._moe_forward(n2)

		return self.normf(out)


def build_rope_cache(L, Dh):
	# Build rotary frequencies for Dh//2 pairs; any odd tail channel is not rotated.
	Dh2 = Dh // 2
	freq = 150000.0 ** ((2.0 * torch.arange(0, Dh2, dtype=torch.float32)) / Dh)
	inv_freq = 1.0 / freq
	t = torch.arange(L, dtype=torch.float32)
	freqs = torch.einsum("i,j->ij", t, inv_freq)
	cos = torch.cos(freqs)
	sin = torch.sin(freqs)
	return cos, sin


def export_ref_weights(model):
	return {
		"Wq": np.ascontiguousarray(model.wq.weight.detach().cpu().numpy().T.astype(np.float32)),
		"bq": np.ascontiguousarray(model.wq.bias.detach().cpu().numpy().astype(np.float32)),
		"Wk": np.ascontiguousarray(model.wk.weight.detach().cpu().numpy().T.astype(np.float32)),
		"bk": np.ascontiguousarray(model.wk.bias.detach().cpu().numpy().astype(np.float32)),
		"Wv": np.ascontiguousarray(model.wv.weight.detach().cpu().numpy().T.astype(np.float32)),
		"bv": np.ascontiguousarray(model.wv.bias.detach().cpu().numpy().astype(np.float32)),
		"Wo": np.ascontiguousarray(model.wo.weight.detach().cpu().numpy().T.astype(np.float32)),
		"bo": np.ascontiguousarray(model.wo.bias.detach().cpu().numpy().astype(np.float32)),
		"Wgate": np.ascontiguousarray(model.wgate.weight.detach().cpu().numpy().T.astype(np.float32)),
		"Wg_experts": np.ascontiguousarray(model.wg_experts.detach().cpu().numpy().astype(np.float32)),
		"Wl_experts": np.ascontiguousarray(model.wl_experts.detach().cpu().numpy().astype(np.float32)),
		"W2_experts": np.ascontiguousarray(model.w2_experts.detach().cpu().numpy().astype(np.float32)),
		"Bg_experts": np.ascontiguousarray(model.bg_experts.detach().cpu().numpy().astype(np.float32)),
		"Bl_experts": np.ascontiguousarray(model.bl_experts.detach().cpu().numpy().astype(np.float32)),
		"B2_experts": np.ascontiguousarray(model.b2_experts.detach().cpu().numpy().astype(np.float32)),
		"norm1_scale": np.ascontiguousarray(model.norm1.scale.detach().cpu().numpy().astype(np.float32)),
		"norm2_scale": np.ascontiguousarray(model.norm2.scale.detach().cpu().numpy().astype(np.float32)),
		"normf_scale": np.ascontiguousarray(model.normf.scale.detach().cpu().numpy().astype(np.float32)),
		"sinks": np.ascontiguousarray(model.sinks.detach().cpu().numpy().astype(np.float32)),
	}


def patch_vitis_kernel_alias_init(project_dir):
	"""Patch generated kernel.cpp for sw_emu robustness.

	1) Fix generated C++ aliases that Vitis HLS rejects.
	Transforms lines like:
	  float v96[4] = A_fifo[0][0];
	into:
	  float *v96 = A_fifo[0][0];

	2) Convert very large local arrays to static storage.
	This avoids oversized stack frames in emulation for large configs.
	"""
	kernel_path = os.path.join(project_dir, "kernel.cpp")
	if not os.path.exists(kernel_path):
		return
	alias_pat = re.compile(
		r"^(\s*)([A-Za-z_]\w*(?:<[^>]+>)?)\s+([A-Za-z_]\w*)\[(\d+)\]\s*=\s*([^;]+);(\s*//.*)?$"
	)
	array_decl_pat = re.compile(
		r"^(\s*)(?!static\b)([A-Za-z_]\w*(?:<[^>]+>)?)\s+([A-Za-z_]\w*)((?:\[\d+\]){1,})\s*;(\s*//.*)?$"
	)
	dtype_bytes = {
		"float": 4,
		"double": 8,
		"int": 4,
		"unsigned": 4,
		"short": 2,
		"unsigned short": 2,
		"int8_t": 1,
		"uint8_t": 1,
		"int16_t": 2,
		"uint16_t": 2,
		"int32_t": 4,
		"uint32_t": 4,
		"int64_t": 8,
		"uint64_t": 8,
	}
	large_array_threshold_bytes = 1 << 20
	with open(kernel_path, "r", encoding="utf-8") as f:
		lines = f.readlines()
	changed = False
	new_lines = []
	for line in lines:
		m = alias_pat.match(line.rstrip("\n"))
		if m:
			indent, dtype, var, _size, rhs, comment = m.groups()
			rhs_stripped = rhs.strip()
			# Keep true initializer lists untouched; only patch alias-like cases.
			if not rhs_stripped.startswith("{") and "[" in rhs_stripped:
				suffix = comment if comment is not None else ""
				line = f"{indent}{dtype} *{var} = {rhs_stripped};{suffix}\n"
				changed = True
		else:
			m_arr = array_decl_pat.match(line.rstrip("\n"))
			if m_arr:
				indent, dtype, var, dims_str, comment = m_arr.groups()
				dims = [int(x) for x in re.findall(r"\[(\d+)\]", dims_str)]
				elems = 1
				for d in dims:
					elems *= d
				elem_bytes = dtype_bytes.get(dtype)
				if elem_bytes is not None and elems * elem_bytes >= large_array_threshold_bytes:
					suffix = comment if comment is not None else ""
					line = f"{indent}static {dtype} {var}{dims_str};{suffix}\n"
					changed = True
		new_lines.append(line)
	if changed:
		with open(kernel_path, "w", encoding="utf-8") as f:
			f.writelines(new_lines)


def check_vitis_runtime_env():
	"""Validate runtime env before launching sw_emu/hw_emu."""
	missing = [name for name in ("XDEVICE", "XILINX_XRT") if name not in os.environ]
	if not missing:
		return
	missing_str = ", ".join(missing)
	raise RuntimeError(
		"Missing Vitis runtime environment variable(s): "
		+ missing_str
		+ ". Use `source /home/gz284/allo_init.sh && python3 gpt-oss-allo.py` "
		+ "(use `&&`, not `&`)."
	)


def load_runtime_config(config_path=None, use_tiny_config=True):
	if use_tiny_config:
		return {
			"hidden_size": 8,
			"num_attention_heads": 4,
			"num_key_value_heads": 2,
			"intermediate_size": 16,
			"vocab_size": 8,
			"sliding_window": 2,
			"num_local_experts": 32,
			"experts_per_token": 4,
		}

	if config_path and os.path.exists(config_path):
		with open(config_path, "r", encoding="utf-8") as f:
			cfg = json.load(f)
		return {
			"hidden_size": int(cfg["hidden_size"]),
			"num_attention_heads": int(cfg["num_attention_heads"]),
			"num_key_value_heads": int(cfg["num_key_value_heads"]),
			"intermediate_size": int(cfg["intermediate_size"]),
			"vocab_size": int(cfg["vocab_size"]),
			"sliding_window": int(cfg.get("sliding_window")),
			"num_local_experts": int(cfg["num_local_experts"]),
			"experts_per_token": int(cfg["experts_per_token"]),
		}
	
	print("no config found, using default tiny config")
	return {
		"hidden_size": 8,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"intermediate_size": 16,
		"vocab_size": 8,
		"sliding_window": 2,
		"num_experts": 32,
		"experts_per_token": 4,
	}

def run_allo_module(mod_inst, x_np, w, cos_np, sin_np, output=None):
	if output is None:
		output = np.zeros_like(x_np, dtype=np.float32)
	if hasattr(mod_inst, "in_types") and len(mod_inst.in_types) == 75:
		B, L, D = x_np.shape
		HQ = w["sinks"].shape[0]
		Dkv = w["Wk"].shape[1]
		Dh = D // HQ
		E = w["Wgate"].shape[1]
		Dffn = w["Wg_experts"].shape[2]
		topk = 4

		n1 = np.zeros((B, L, D), dtype=np.float32)
		Q = np.zeros((B, L, D), dtype=np.float32)
		K_arr = np.zeros((B, L, Dkv), dtype=np.float32)
		V_arr = np.zeros((B, L, Dkv), dtype=np.float32)
		Q_rot = np.zeros((B * HQ, L, Dh), dtype=np.float32)
		K_rot = np.zeros((B * HQ, Dh, L), dtype=np.float32)
		V_h = np.zeros((B * HQ, L, Dh), dtype=np.float32)
		scores = np.zeros((B * HQ, L, L), dtype=np.float32)
		probs = np.zeros((B * HQ, L, L), dtype=np.float32)
		attn_ctx_h = np.zeros((B * HQ, L, Dh), dtype=np.float32)
		attn_ctx = np.zeros((B, L, D), dtype=np.float32)
		O = np.zeros((B, L, D), dtype=np.float32)
		res1 = np.zeros((B, L, D), dtype=np.float32)
		n2 = np.zeros((B, L, D), dtype=np.float32)
		n2_flat = np.zeros((B * L, 1, D), dtype=np.float32)
		gate_logits = np.zeros((B * L, 1, E), dtype=np.float32)
		expert_idx = np.zeros((B * L * topk,), dtype=np.int32)
		expert_ws = np.zeros((B * L * topk,), dtype=np.float32)
		moe_g_in = np.zeros((B * L * topk, 1, D), dtype=np.float32)
		moe_g_w = np.zeros((B * L * topk, D, Dffn), dtype=np.float32)
		moe_l_w = np.zeros((B * L * topk, D, Dffn), dtype=np.float32)
		moe_g_out = np.zeros((B * L * topk, 1, Dffn), dtype=np.float32)
		moe_l_out = np.zeros((B * L * topk, 1, Dffn), dtype=np.float32)
		moe_down_in = np.zeros((B * L * topk, 1, Dffn), dtype=np.float32)
		moe_down_w = np.zeros((B * L * topk, Dffn, D), dtype=np.float32)
		moe_down_out = np.zeros((B * L * topk, 1, D), dtype=np.float32)
		moe_reduce_out = np.zeros((B, L, D), dtype=np.float32)

		args = [
			x_np,
			w["norm1_scale"],
			n1,
			n1,
			w["Wq"],
			Q,
			n1,
			w["Wk"],
			K_arr,
			n1,
			w["Wv"],
			V_arr,
			Q,
			w["bq"],
			K_arr,
			w["bk"],
			V_arr,
			w["bv"],
			cos_np,
			sin_np,
			Q_rot,
			K_rot,
			V_h,
			Q_rot,
			K_rot,
			scores,
			scores,
			w["sinks"],
			probs,
			probs,
			V_h,
			attn_ctx_h,
			attn_ctx_h,
			attn_ctx,
			attn_ctx,
			w["Wo"],
			O,
			O,
			w["bo"],
			w["norm2_scale"],
			res1,
			n2,
			n2_flat,
			moe_g_in,
			n2,
			w["Wgate"],
			gate_logits,
			gate_logits,
			expert_idx,
			expert_ws,
			w["Wg_experts"],
			w["Wl_experts"],
			moe_g_w,
			moe_l_w,
			moe_g_in,
			moe_g_w,
			moe_g_out,
			moe_g_in,
			moe_l_w,
			moe_l_out,
			moe_g_out,
			moe_l_out,
			w["Bg_experts"],
			w["Bl_experts"],
			w["W2_experts"],
			moe_down_in,
			moe_down_w,
			moe_down_in,
			moe_down_w,
			moe_down_out,
			moe_down_out,
			w["B2_experts"],
			moe_reduce_out,
			w["normf_scale"],
			output,
		]
		mod_inst(*args)
		return output
	args = [
		x_np,
		w["Wq"],
		w["bq"],
		w["Wk"],
		w["bk"],
		w["Wv"],
		w["bv"],
		w["Wo"],
		w["bo"],
		w["Wgate"],
		w["Wg_experts"],
		w["Wl_experts"],
		w["W2_experts"],
		w["Bg_experts"],
		w["Bl_experts"],
		w["B2_experts"],
		w["norm1_scale"],
		w["norm2_scale"],
		w["normf_scale"],
		w["sinks"],
		cos_np,
		sin_np,
		output,
	]
	mod_inst(*args)
	return output


def run_mode(
	B,
	L,
	D,
	HQ,
	HKV,
	Qm,
	Dkv,
	Dh,
	Dh2,
	Dffn,
	E,
	Vocab,
	Win,
	K,
	Mt,
	Nt,
	vitis_device,
	run_sw_emu,
	run_hw_emu
):
	ref = TinyGptOssRef(D, HQ, HKV, Dffn, Vocab, Win, E, K).eval()
	cos_t, sin_t = build_rope_cache(L, Dh)

	x_t = torch.rand(B, L, D)
	golden = ref(x_t, cos_t, sin_t).detach().cpu().numpy().astype(np.float32)

	w = export_ref_weights(ref)
	x_np = np.ascontiguousarray(x_t.detach().cpu().numpy().astype(np.float32))
	cos_np = np.ascontiguousarray(cos_t.detach().cpu().numpy().astype(np.float32))
	sin_np = np.ascontiguousarray(sin_t.detach().cpu().numpy().astype(np.float32))
	# tiles = build_systolic_tiles(Mt, Nt)
	# if Dh % 2 != 0 and (tiles["QKT"] != (1, 1) or tiles["YV"] != (1, 1)):
	# 	print(
	# 		f"odd Dh={Dh} detected; overriding QKT/YV tiles from {tiles['QKT']}/{tiles['YV']} to (1, 1) for numeric parity"
	# 	)
	# 	tiles["QKT"] = (1, 1)
	# 	tiles["YV"] = (1, 1)
	# if tiles["MOE_GATE"][0] != 1 or tiles["MOE_G"][0] != 1 or tiles["MOE_L"][0] != 1 or tiles["MOE_O"][0] != 1:
	# 	raise ValueError("MOE systolic arrays have M=1, so their Mt tile must be 1.")
	s = build_gpt_oss_schedule(B, HQ, HKV, Qm, L, D, Dkv, Dh, Dh2, Dffn, E, Vocab, Win)
	mod = s.build(target="llvm")
	allo_out = run_allo_module(mod, x_np, w, cos_np, sin_np)

	np.testing.assert_allclose(allo_out, golden, atol=3e-2, rtol=3e-2)
	print("Output matches PyTorch reference.")

	if run_sw_emu or run_hw_emu:
		if not hls.is_available("vitis_hls"):
			raise RuntimeError("vitis_hls is not available, cannot run sw_emu/hw_emu")
		check_vitis_runtime_env()

		# Do not force OCL_ICD_VENDORS here: on some hosts it can hide Xilinx platform
		# discovery in emulation and trigger CL_PLATFORM_NOT_FOUND_KHR (-1001).
		if "OCL_ICD_FILENAMES" not in os.environ:
			xilinx_opencl = "/opt/xilinx/xrt/lib/libxilinxopencl.so"
			if os.path.exists(xilinx_opencl):
				os.environ["OCL_ICD_FILENAMES"] = xilinx_opencl

		if run_sw_emu:
			print("Running sw_emu")
			sw_project = "gpt_oss_sw_emu"
			if os.path.isdir(sw_project):
				shutil.rmtree(sw_project)
			mod_sw = s.build(
				target="vitis_hls",
				mode="sw_emu",
				project=sw_project,
				configs={"device": vitis_device},
			)
			patch_vitis_kernel_alias_init(mod_sw.project)
			out_sw = np.zeros_like(golden, dtype=np.float32)
			run_allo_module(mod_sw, x_np, w, cos_np, sin_np, output=out_sw)
			np.testing.assert_allclose(out_sw, golden, atol=3e-2, rtol=3e-2)
			print("sw_emu matches PyTorch reference.")

		if run_hw_emu:
			print("Running hw_emu ")
			hw_project = "gpt_oss_hw_emu"
			if os.path.isdir(hw_project):
				shutil.rmtree(hw_project)
			mod_hw = s.build(
				target="vitis_hls",
				mode="hw_emu",
				project=hw_project,
				configs={"device": vitis_device},
			)
			patch_vitis_kernel_alias_init(mod_hw.project)
			out_hw = np.zeros_like(golden, dtype=np.float32)
			run_allo_module(mod_hw, x_np, w, cos_np, sin_np, output=out_hw)
			np.testing.assert_allclose(out_hw, golden, atol=3e-2, rtol=3e-2)
			print("hw_emu matches PyTorch reference.")

			report_path = _find_csynth_report(hw_project)
			if report_path is None:
				print("No csynth report found under hw_emu project.")
			else:
				cycles = _parse_latency_cycles_from_csynth(report_path)
				if cycles is None:
					print(f"Found report but failed to parse latency cycles: {report_path}")
				else:
					theory_macs = estimate_theoretical_macs(L, D, Dkv, Dh, HQ, E, Dffn, K)
					effective_macs = theory_macs / cycles if cycles > 0 else 0.0
					print(f"[HLS] report: {report_path}")
					print(f"[HLS] latency cycles: {cycles}")
					print(f"[HLS] theoretical MACs (per block): {theory_macs}")
					print(f"[HLS] effective MACs/cycle: {effective_macs:.6f}")


def main(
	config_path="/home/gz284/allo/my_part/config.json",
	use_tiny_config=True,
	batch_size=1,
	seq_len=32,
	mt=1,
	nt=1,
	vitis_device="u280",
	run_sw_emu=False,
	run_hw_emu=False,
):
	torch.manual_seed(0)
	np.random.seed(0)

	cfg = load_runtime_config(config_path=config_path, use_tiny_config=use_tiny_config)
	B = batch_size
	L = seq_len
	D = cfg["hidden_size"]
	HQ = cfg["num_attention_heads"]
	HKV = cfg["num_key_value_heads"]
	Qm = HQ // HKV
	Dh = D // HQ
	Dh2 = Dh // 2
	Dkv = HKV * Dh
	Dffn = cfg["intermediate_size"]
	Vocab = cfg["vocab_size"]
	Win = cfg["sliding_window"]
	E = cfg["num_local_experts"]
	K = min(4, cfg["experts_per_token"])
	Mt = mt
	Nt = nt
	print(f"B:{B}, L:{L}, D:{D}, HQ:{HQ}, HKV:{HKV}, Qm:{Qm}, Dkv:{Dkv}, Dh:{Dh}, Dh2:{Dh2}, Dffn:{Dffn}, E:{E}, Vocab:{Vocab}, Win:{Win}, K:{K}, Mt:{Mt}, Nt:{Nt}")
	run_mode(
		B,
		L,
		D,
		HQ,
		HKV,
		Qm,
		Dkv,
		Dh,
		Dh2,
		Dffn,
		E,
		Vocab,
		Win,
		K,
		Mt,
		Nt,
		vitis_device,
		run_sw_emu=run_sw_emu,
		run_hw_emu=run_hw_emu,
	)


if __name__ == "__main__":
	os.environ.setdefault("OMP_NUM_THREADS", "128")
	main()
