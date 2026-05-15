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
from allo.backend import hls
from allo.ir.types import float32
from allo.library.systolic import schedule_systolic, systolic


def add2d[Ty, L, D](A: "Ty[L, D]", B: "Ty[L, D]") -> "Ty[L, D]":
	Z: Ty[L, D]
	for i, j in allo.grid(L, D, name="add2d"):
		Z[i, j] = A[i, j] + B[i, j]
	return Z


def bias2d[Ty, L, D](X: "Ty[L, D]", b: "Ty[D]") -> "Ty[L, D]":
	Z: Ty[L, D]
	for i, j in allo.grid(L, D, name="bias2d"):
		Z[i, j] = X[i, j] + b[j]
	return Z


def rmsnorm2d[Ty, L, D](X: "Ty[L, D]", scale: "Ty[D]") -> "Ty[L, D]":
	Z: Ty[L, D]
	mean_sq: Ty[L] = 0.0
	inv_rms: Ty[L]

	for i, j in allo.grid(L, D, name="rms_sum"):
		mean_sq[i] += X[i, j] * X[i, j]

	for i in allo.grid(L, name="rms_inv"):
		mean_sq[i] = mean_sq[i] / float(D)
		inv_rms[i] = 1.0 / allo.sqrt(mean_sq[i] + 0.00001)

	for i, j in allo.grid(L, D, name="rms_apply"):
		Z[i, j] = X[i, j] * inv_rms[i] * scale[j]

	return Z


def swiglu2d[Ty, L, D](X_glu: "Ty[L, D]", X_linear: "Ty[L, D]") -> "Ty[L, D]":
	Z: Ty[L, D]
	alpha: Ty = 1.702
	limit: Ty = 7.0
	one: Ty = 1.0

	for i, j in allo.grid(L, D, name="swiglu2d"):
		xg: Ty = X_glu[i, j]
		xl: Ty = X_linear[i, j]

		if xg > limit:
			xg = limit
		if xl > limit:
			xl = limit
		if xl < -limit:
			xl = -limit

		sig: Ty = one / (one + allo.exp(-alpha * xg))
		Z[i, j] = (xg * sig) * (xl + one)

	return Z


def moe_layer_topk_map4[Ty, L, D, Dffn, E, MtGate, NtGate, MtG, NtG, MtL, NtL, MtDown, NtDown](
	X: "Ty[L, D]",
	Wgate: "Ty[D, E]",
	Wg_experts: "Ty[E, D, Dffn]",
	Wl_experts: "Ty[E, D, Dffn]",
	W2_experts: "Ty[E, Dffn, D]",
	Bg_experts: "Ty[E, Dffn]",
	Bl_experts: "Ty[E, Dffn]",
	B2_experts: "Ty[E, D]",
) -> "Ty[L, D]":
	Y: Ty[L, D] = 0.0
	alpha: Ty = 1.702
	limit: Ty = 7.0
	one: Ty = 1.0
	neg_inf: Ty = -1000000000.0

	for i in range(L):
		X_row: Ty[1, D]
		gate_logits: Ty[1, E] = 0.0
		for d in range(D):
			X_row[0, d] = X[i, d]

		systolic[Ty, Ty, Ty, 1, D, E, MtGate, NtGate, "MOE_GATE"](X_row, Wgate, gate_logits)

		i0 = 0
		i1 = 0
		i2 = 0
		i3 = 0
		v0: Ty = neg_inf
		v1: Ty = neg_inf
		v2: Ty = neg_inf
		v3: Ty = neg_inf

		for e in range(E):
			v: Ty = gate_logits[0, e]

			if v > v0:
				v3 = v2
				i3 = i2
				v2 = v1
				i2 = i1
				v1 = v0
				i1 = i0
				v0 = v
				i0 = e
			elif v > v1:
				v3 = v2
				i3 = i2
				v2 = v1
				i2 = i1
				v1 = v
				i1 = e
			elif v > v2:
				v3 = v2
				i3 = i2
				v2 = v
				i2 = e
			elif v > v3:
				v3 = v
				i3 = e

		m: Ty = v0
		w0: Ty = allo.exp(v0 - m)
		w1: Ty = allo.exp(v1 - m)
		w2: Ty = allo.exp(v2 - m)
		w3: Ty = allo.exp(v3 - m)
		sumw: Ty = w0 + w1 + w2 + w3
		w0 = w0 / sumw
		w1 = w1 / sumw
		w2 = w2 / sumw
		w3 = w3 / sumw

		for s in range(4):
			e = 0
			ws: Ty = 0.0
			if s == 0:
				e = i0
				ws = w0
			elif s == 1:
				e = i1
				ws = w1
			elif s == 2:
				e = i2
				ws = w2
			else:
				e = i3
				ws = w3
			Wg_e: Ty[D, Dffn]
			Wl_e: Ty[D, Dffn]
			for d, f in allo.grid(D, Dffn, name="moe_copy_up"):
				Wg_e[d, f] = Wg_experts[e, d, f]
				Wl_e[d, f] = Wl_experts[e, d, f]

			hidden_g: Ty[1, Dffn] = 0.0
			hidden_l: Ty[1, Dffn] = 0.0
			systolic[Ty, Ty, Ty, 1, D, Dffn, MtG, NtG, "MOE_G"](X_row, Wg_e, hidden_g)
			systolic[Ty, Ty, Ty, 1, D, Dffn, MtL, NtL, "MOE_L"](X_row, Wl_e, hidden_l)

			hidden: Ty[Dffn]
			for f in range(Dffn):
				xg: Ty = hidden_g[0, f] + Bg_experts[e, f]
				xl: Ty = hidden_l[0, f] + Bl_experts[e, f]
				if xg > limit:
					xg = limit
				if xl > limit:
					xl = limit
				if xl < -limit:
					xl = -limit
				sig: Ty = one / (one + allo.exp(-alpha * xg))
				hidden[f] = (xg * sig) * (xl + one)

			H_row: Ty[1, Dffn]
			for f in range(Dffn):
				H_row[0, f] = hidden[f]

			W2_e: Ty[Dffn, D]
			for f, d in allo.grid(Dffn, D, name="moe_copy_down"):
				W2_e[f, d] = W2_experts[e, f, d]

			out_row: Ty[1, D] = 0.0
			systolic[Ty, Ty, Ty, 1, Dffn, D, MtDown, NtDown, "MOE_O"](H_row, W2_e, out_row)
			for d in range(D):
				Y[i, d] += ws * (out_row[0, d] + B2_experts[e, d])

	return Y


def softmax_causal_sinks2d[Ty, L, Win](X: "Ty[L, L]", sink: Ty) -> "Ty[L, L]":
	Z: Ty[L, L]
	E: Ty[L, L]
	M: Ty[L] = -1000000000.0
	Es: Ty[L]
	S: Ty[L] = 0.0

	for i, j in allo.grid(L, L, name="smx_row_max"):
		v: Ty = X[i, j]
		if j > i:
			v = -1000000000.0
		if Win > 0:
			if j + Win <= i:
				v = -1000000000.0
		if v > M[i]:
			M[i] = v

	for i in allo.grid(L, name="smx_sink_max"):
		if sink > M[i]:
			M[i] = sink

	for i, j in allo.grid(L, L, name="smx_exp_sum"):
		v: Ty = X[i, j]
		if j > i:
			v = -1000000000.0
		if Win > 0:
			if j + Win <= i:
				v = -1000000000.0
		E[i, j] = allo.exp(v - M[i])
		S[i] += E[i, j]

	for i in allo.grid(L, name="smx_sink_sum"):
		Es[i] = allo.exp(sink - M[i])
		S[i] += Es[i]

	for i, j in allo.grid(L, L, name="smx_update"):
		Z[i, j] = E[i, j] / S[i]

	return Z


def rotary_apply_head[Ty, L, Dh, Dh2](X_h: "Ty[L, Dh]", cos: "Ty[L, Dh2]", sin: "Ty[L, Dh2]") -> "Ty[L, Dh]":
	Z_h: Ty[L, Dh]
	for i, j in allo.grid(L, Dh2, name="rope_apply"):
		x1: Ty = X_h[i, j]
		x2: Ty = X_h[i, j + Dh2]
		c: Ty = cos[i, j]
		s: Ty = sin[i, j]
		Z_h[i, j] = x1 * c - x2 * s
		Z_h[i, j + Dh2] = x2 * c + x1 * s

	# If Dh is odd, keep the final unpaired channel unchanged.
	if Dh2 * 2 < Dh:
		for i in allo.grid(L, name="rope_tail_copy"):
			Z_h[i, Dh - 1] = X_h[i, Dh - 1]
	return Z_h


def causal_attention_head[Ty, L, Dh, Win, MtQKT, NtQKT, MtYV, NtYV](
	Q_h: "Ty[L, Dh]", K_h: "Ty[L, Dh]", V_h: "Ty[L, Dh]", sink: Ty
) -> "Ty[L, Dh]":
	K_t: Ty[Dh, L]
	scores: Ty[L, L] = 0.0
	out_h: Ty[L, Dh] = 0.0

	for i, j in allo.grid(L, Dh, name="k_transpose"):
		K_t[j, i] = K_h[i, j]

	systolic[Ty, Ty, Ty, L, Dh, L, MtQKT, NtQKT, "QKT"](Q_h, K_t, scores)

	for i, j in allo.grid(L, L, name="scale_scores"):
		scores[i, j] = scores[i, j] / allo.sqrt(float(Dh))

	probs = softmax_causal_sinks2d[Ty, L, Win](scores, sink)
	systolic[Ty, Ty, Ty, L, L, Dh, MtYV, NtYV, "YV"](probs, V_h, out_h)
	return out_h


def gpt_oss_block[
	Ty,
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
	Win,
	MtQ,
	NtQ,
	MtK,
	NtK,
	MtV,
	NtV,
	MtO,
	NtO,
	MtQKT,
	NtQKT,
	MtYV,
	NtYV,
	MtMoeGate,
	NtMoeGate,
	MtMoeG,
	NtMoeG,
	MtMoeL,
	NtMoeL,
	MtMoeDown,
	NtMoeDown,
](
	X: "Ty[L, D]",
	Wq: "Ty[D, D]",
	bq: "Ty[D]",
	Wk: "Ty[D, Dkv]",
	bk: "Ty[Dkv]",
	Wv: "Ty[D, Dkv]",
	bv: "Ty[Dkv]",
	Wo: "Ty[D, D]",
	bo: "Ty[D]",
	Wgate: "Ty[D, E]",
	Wg_experts: "Ty[E, D, Dffn]",
	Wl_experts: "Ty[E, D, Dffn]",
	W2_experts: "Ty[E, Dffn, D]",
	Bg_experts: "Ty[E, Dffn]",
	Bl_experts: "Ty[E, Dffn]",
	B2_experts: "Ty[E, D]",
	norm1_scale: "Ty[D]",
	norm2_scale: "Ty[D]",
	normf_scale: "Ty[D]",
	sinks: "Ty[HQ]",
	cos: "Ty[L, Dh2]",
	sin: "Ty[L, Dh2]",
) -> "Ty[L, D]":
	n1 = rmsnorm2d[Ty, L, D](X, norm1_scale)

	Q: Ty[L, D] = 0.0
	K: Ty[L, Dkv] = 0.0
	V: Ty[L, Dkv] = 0.0
	attn_ctx: Ty[L, D]

	systolic[Ty, Ty, Ty, L, D, D, MtQ, NtQ, "Q"](n1, Wq, Q)
	systolic[Ty, Ty, Ty, L, D, Dkv, MtK, NtK, "K"](n1, Wk, K)
	systolic[Ty, Ty, Ty, L, D, Dkv, MtV, NtV, "V"](n1, Wv, V)

	Q = bias2d[Ty, L, D](Q, bq)
	for i, j in allo.grid(L, Dkv, name="bias_k"):
		K[i, j] = K[i, j] + bk[j]
	for i, j in allo.grid(L, Dkv, name="bias_v"):
		V[i, j] = V[i, j] + bv[j]

	for kv in range(HKV):
		K_h: Ty[L, Dh]
		V_h: Ty[L, Dh]

		for i, j in allo.grid(L, Dh, name="mha_split"):
			K_h[i, j] = K[i, kv * Dh + j]
			V_h[i, j] = V[i, kv * Dh + j]

		K_r = rotary_apply_head[Ty, L, Dh, Dh2](K_h, cos, sin)

		for qm in range(Qm):
			h = kv * Qm + qm
			Q_h: Ty[L, Dh]
			for i, j in allo.grid(L, Dh, name="mha_split_q"):
				Q_h[i, j] = Q[i, h * Dh + j]

			Q_r = rotary_apply_head[Ty, L, Dh, Dh2](Q_h, cos, sin)
			C_h = causal_attention_head[Ty, L, Dh, Win, MtQKT, NtQKT, MtYV, NtYV](Q_r, K_r, V_h, sinks[h])

			for i, j in allo.grid(L, Dh, name="mha_merge"):
				attn_ctx[i, h * Dh + j] = C_h[i, j]

	O: Ty[L, D] = 0.0
	systolic[Ty, Ty, Ty, L, D, D, MtO, NtO, "O"](attn_ctx, Wo, O)
	O = bias2d[Ty, L, D](O, bo)

	res1 = add2d[Ty, L, D](X, O)
	n2 = rmsnorm2d[Ty, L, D](res1, norm2_scale)

	out_ffn = moe_layer_topk_map4[
		Ty,
		L,
		D,
		Dffn,
		E,
		MtMoeGate,
		NtMoeGate,
		MtMoeG,
		NtMoeG,
		MtMoeL,
		NtMoeL,
		MtMoeDown,
		NtMoeDown,
	](
		n2, Wgate, Wg_experts, Wl_experts, W2_experts, Bg_experts, Bl_experts, B2_experts
	)

	out = add2d[Ty, L, D](res1, out_ffn)
	return rmsnorm2d[Ty, L, D](out, normf_scale)


def gpt_oss_dataflow[
	Ty,
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
	MtQ,
	NtQ,
	MtK,
	NtK,
	MtV,
	NtV,
	MtO,
	NtO,
	MtQKT,
	NtQKT,
	MtYV,
	NtYV,
	MtMoeGate,
	NtMoeGate,
	MtMoeG,
	NtMoeG,
	MtMoeL,
	NtMoeL,
	MtMoeDown,
	NtMoeDown,
](
	X: "Ty[B, L, D]",
	Wq: "Ty[D, D]",
	bq: "Ty[D]",
	Wk: "Ty[D, Dkv]",
	bk: "Ty[Dkv]",
	Wv: "Ty[D, Dkv]",
	bv: "Ty[Dkv]",
	Wo: "Ty[D, D]",
	bo: "Ty[D]",
	Wgate: "Ty[D, E]",
	Wg_experts: "Ty[E, D, Dffn]",
	Wl_experts: "Ty[E, D, Dffn]",
	W2_experts: "Ty[E, Dffn, D]",
	Bg_experts: "Ty[E, Dffn]",
	Bl_experts: "Ty[E, Dffn]",
	B2_experts: "Ty[E, D]",
	norm1_scale: "Ty[D]",
	norm2_scale: "Ty[D]",
	normf_scale: "Ty[D]",
	sinks: "Ty[HQ]",
	cos: "Ty[L, Dh2]",
	sin: "Ty[L, Dh2]",
) -> "Ty[B, L, D]":
	Y: Ty[B, L, D]

	for b in range(B):
		X_b: Ty[L, D]
		for i, j in allo.grid(L, D, name="copy_batch"):
			X_b[i, j] = X[b, i, j]

		Y_b = gpt_oss_block[
			Ty,
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
			Win,
			MtQ,
			NtQ,
			MtK,
			NtK,
			MtV,
			NtV,
			MtO,
			NtO,
			MtQKT,
			NtQKT,
			MtYV,
			NtYV,
			MtMoeGate,
			NtMoeGate,
			MtMoeG,
			NtMoeG,
			MtMoeL,
			NtMoeL,
			MtMoeDown,
			NtMoeDown,
		](
			X_b,
			Wq,
			bq,
			Wk,
			bk,
			Wv,
			bv,
			Wo,
			bo,
			Wgate,
			Wg_experts,
			Wl_experts,
			W2_experts,
			Bg_experts,
			Bl_experts,
			B2_experts,
			norm1_scale,
			norm2_scale,
			normf_scale,
			sinks,
			cos,
			sin,
		)

		for i, j in allo.grid(L, D, name="copy_hidden"):
			Y[b, i, j] = Y_b[i, j]

	return Y


def make_library_systolic_sch(m, k, n, mt, nt):
	s = allo.customize(systolic, instantiate=[float32, float32, float32, m, k, n, mt, nt])
	s = schedule_systolic(s)
	# Keep the library schedule, but skip stream-conversion primitives during compose.
	# They currently cause PE kernel call type mismatches in this composed graph.
	s.primitive_sequences = [prim for prim in s.primitive_sequences if prim[0] != "to"]
	return s


def build_systolic_tiles(default_mt, default_nt):
	# Centralized systolic tile config. Change per-id Mt/Nt here.
	
	# tiles = {
	# 	"Q": (4, 4),  # max: Mt=L, Nt=D
	# 	"K": (4, 4),  # max: Mt=L, Nt=Dkv
	# 	"V": (4, 4),  # max: Mt=L, Nt=Dkv
	# 	"O": (4, 4),  # max: Mt=L, Nt=D
	# 	"QKT": (8, 4),  # max: Mt=L, Nt=L
	# 	"YV": (8, 4),  # max: Mt=L, Nt=Dh
	# 	"MOE_GATE": (1, 4),  # max: Mt=1, Nt=E
	# 	"MOE_G": (1, 64),  # max: Mt=1, Nt=Dffn
	# 	"MOE_L": (1, 64),  # max: Mt=1, Nt=Dffn
	# 	"MOE_O": (1, 64),  # max: Mt=1, Nt=D
	# }
	tiles = {
		"Q": (1, 1),  # max: Mt=L, Nt=D
		"K": (1, 1),  # max: Mt=L, Nt=Dkv
		"V": (1, 1),  # max: Mt=L, Nt=Dkv
		"O": (1, 1),  # max: Mt=L, Nt=D
		"QKT": (1, 1),  # max: Mt=L, Nt=L
		"YV": (1, 1),  # max: Mt=L, Nt=Dh
		"MOE_GATE": (1, 1),  # max: Mt=1, Nt=E
		"MOE_G": (1, 1),  # max: Mt=1, Nt=Dffn
		"MOE_L": (1, 1),  # max: Mt=1, Nt=Dffn
		"MOE_O": (1, 1),  # max: Mt=1, Nt=D
	}

	# Make command-line mt/nt effective for attention systolic arrays.
	# MOE arrays keep their hand-tuned defaults because Mt must stay 1.
	if default_mt is not None and default_nt is not None and (int(default_mt), int(default_nt)) != (1, 1):
		mt = max(1, int(default_mt))
		nt = max(1, int(default_nt))
		for key in ("Q", "K", "V", "O", "QKT", "YV"):
			tiles[key] = (mt, nt)

	return tiles


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


def build_gpt_oss_schedule(B, HQ, HKV, Qm, L, D, Dkv, Dh, Dh2, Dffn, E, Vocab, Win, tiles):
	q_mt, q_nt = tiles["Q"]
	k_mt, k_nt = tiles["K"]
	v_mt, v_nt = tiles["V"]
	o_mt, o_nt = tiles["O"]
	qkt_mt, qkt_nt = tiles["QKT"]
	yv_mt, yv_nt = tiles["YV"]
	moe_gate_mt, moe_gate_nt = tiles["MOE_GATE"]
	moe_g_mt, moe_g_nt = tiles["MOE_G"]
	moe_l_mt, moe_l_nt = tiles["MOE_L"]
	moe_down_mt, moe_down_nt = tiles["MOE_O"]

	s = allo.customize(
		gpt_oss_dataflow,
		instantiate=[
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
			q_mt,
			q_nt,
			k_mt,
			k_nt,
			v_mt,
			v_nt,
			o_mt,
			o_nt,
			qkt_mt,
			qkt_nt,
			yv_mt,
			yv_nt,
			moe_gate_mt,
			moe_gate_nt,
			moe_g_mt,
			moe_g_nt,
			moe_l_mt,
			moe_l_nt,
			moe_down_mt,
			moe_down_nt,
		],
	)

	s.compose(make_library_systolic_sch(L, D, D, q_mt, q_nt), id="Q")
	s.compose(make_library_systolic_sch(L, D, Dkv, k_mt, k_nt), id="K")
	s.compose(make_library_systolic_sch(L, D, Dkv, v_mt, v_nt), id="V")
	s.compose(make_library_systolic_sch(L, D, D, o_mt, o_nt), id="O")
	s.compose(make_library_systolic_sch(L, Dh, L, qkt_mt, qkt_nt), id="QKT")
	s.compose(make_library_systolic_sch(L, L, Dh, yv_mt, yv_nt), id="YV")
	s.compose(make_library_systolic_sch(1, D, E, moe_gate_mt, moe_gate_nt), id="MOE_GATE")
	s.compose(make_library_systolic_sch(1, D, Dffn, moe_g_mt, moe_g_nt), id="MOE_G")
	s.compose(make_library_systolic_sch(1, D, Dffn, moe_l_mt, moe_l_nt), id="MOE_L")
	s.compose(make_library_systolic_sch(1, Dffn, D, moe_down_mt, moe_down_nt), id="MOE_O")
	return s


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
	]
	if output is not None:
		args.append(output)
	res = mod_inst(*args)
	return output if output is not None else res


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
	tiles = build_systolic_tiles(Mt, Nt)
	if Dh % 2 != 0 and (tiles["QKT"] != (1, 1) or tiles["YV"] != (1, 1)):
		print(
			f"odd Dh={Dh} detected; overriding QKT/YV tiles from {tiles['QKT']}/{tiles['YV']} to (1, 1) for numeric parity"
		)
		tiles["QKT"] = (1, 1)
		tiles["YV"] = (1, 1)
	if tiles["MOE_GATE"][0] != 1 or tiles["MOE_G"][0] != 1 or tiles["MOE_L"][0] != 1 or tiles["MOE_O"][0] != 1:
		raise ValueError("MOE systolic arrays have M=1, so their Mt tile must be 1.")
	s = build_gpt_oss_schedule(B, HQ, HKV, Qm, L, D, Dkv, Dh, Dh2, Dffn, E, Vocab, Win, tiles)
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
	use_tiny_config=False,
	batch_size=1,
	seq_len=32,
	mt=1,
	nt=1,
	vitis_device="u280",
	run_sw_emu=False,
	run_hw_emu=True,
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
	main()
