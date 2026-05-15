import math
import os
import re

import allo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allo.backend import hls
from allo.ir.types import float32
from allo.library.systolic import systolic


class GPT2(nn.Module):
	def __init__(self, vocab_size, n_embd, n_head, n_layers):
		super().__init__()
		self.transformer_blocks = nn.ModuleList(
			[TransformerBlock(n_embd, n_head, n_embd * 4) for _ in range(n_layers)]
		)
		self.ln_f = nn.LayerNorm(n_embd)
		self.fc = nn.Linear(n_embd, vocab_size)

	def forward(self, x):
		for block in self.transformer_blocks:
			x = block(x)
		x = self.ln_f(x)
		x = self.fc(x)
		return x


class TransformerBlock(nn.Module):
	def __init__(self, n_embd, num_heads, ffn_hidden_dim):
		super().__init__()
		self.attention = MultiHeadAttention(n_embd, num_heads)
		self.norm1 = nn.LayerNorm(n_embd)
		self.ffn = FFN(n_embd, ffn_hidden_dim, n_embd)
		self.norm2 = nn.LayerNorm(n_embd)

	def forward(self, x):
		attn_output = self.attention(x)
		out1 = self.norm1(x + attn_output)
		ffn_output = self.ffn(out1)
		out2 = self.norm2(out1 + ffn_output)
		return out2


class FFN(nn.Module):
	def __init__(self, n_embd, hidden_dim, output_dim):
		super().__init__()
		self.fc1 = nn.Linear(n_embd, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)
		self.activation = nn.GELU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.activation(x)
		x = self.fc2(x)
		return x


class MultiHeadAttention(nn.Module):
	def __init__(self, n_embd, num_heads):
		super().__init__()
		self.num_heads = num_heads
		self.head_dim = n_embd // num_heads
		self.linear_q = nn.Linear(n_embd, n_embd)
		self.linear_k = nn.Linear(n_embd, n_embd)
		self.linear_v = nn.Linear(n_embd, n_embd)
		self.linear_out = nn.Linear(n_embd, n_embd)

	def mask(self, x):
		ones = torch.ones(x.size(1), x.size(1), device=x.device)
		causal_mask = (1 - torch.tril(ones)) * -1e10
		return causal_mask

	def split_heads(self, x):
		new_shape = x.shape[:-1] + (self.num_heads, -1)
		x = x.view(new_shape)
		return x.permute(0, 2, 1, 3)

	def scaled_dot_product(self, q, k, v, x):
		attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
		attn_score = attn_score + self.mask(x)
		attn_probs = F.softmax(attn_score, dim=-1)
		attn = torch.matmul(attn_probs, v)
		return attn

	def forward(self, x):
		q = self.split_heads(self.linear_q(x))
		k = self.split_heads(self.linear_k(x))
		v = self.split_heads(self.linear_v(x))
		output = self.scaled_dot_product(q, k, v, x)
		output = output.permute(0, 2, 1, 3)
		output = output.reshape(output.shape[0], output.shape[1], -1)
		output = self.linear_out(output)
		return output


def gpt_add2d[Ty, L, D](A: "Ty[L, D]", B: "Ty[L, D]") -> "Ty[L, D]":
	Z: Ty[L, D]
	for i, j in allo.grid(L, D, name="add2d"):
		Z[i, j] = A[i, j] + B[i, j]
	return Z


def gpt_bias2d[Ty, L, D](X: "Ty[L, D]", b: "Ty[D]") -> "Ty[L, D]":
	Z: Ty[L, D]
	for i, j in allo.grid(L, D, name="bias2d"):
		Z[i, j] = X[i, j] + b[j]
	return Z


def gpt_gelu2d[Ty, L, D](X: "Ty[L, D]") -> "Ty[L, D]":
	Z: Ty[L, D]
	for i, j in allo.grid(L, D, name="gelu2d"):
		x: Ty = X[i, j]
		Z[i, j] = 0.5 * x * (1.0 + allo.tanh(0.797885 * (x + 0.044715 * allo.power(x, 3.0))))
	return Z


def gpt_layer_norm2d[Ty, L, D](X: "Ty[L, D]", gamma: "Ty[D]", beta: "Ty[D]") -> "Ty[L, D]":
	Z: Ty[L, D]
	mean: Ty[L] = 0.0
	mean2: Ty[L] = 0.0
	var: Ty[L]

	for i, j in allo.grid(L, D, name="ln_sum"):
		mean[i] += X[i, j]
		mean2[i] += X[i, j] * X[i, j]

	for i in allo.grid(L, name="ln_mean_var"):
		mean[i] = mean[i] / float(D)
		mean2[i] = mean2[i] / float(D)
		var[i] = mean2[i] - mean[i] * mean[i]

	for i, j in allo.grid(L, D, name="ln_norm"):
		Z[i, j] = gamma[j] * (X[i, j] - mean[i]) / allo.sqrt(var[i] + 0.00001) + beta[j]

	return Z


def gpt_softmax_causal[Ty, L](X: "Ty[L, L]") -> "Ty[L, L]":
	Z: Ty[L, L]
	E: Ty[L, L]
	M: Ty[L] = -1000000000.0
	S: Ty[L] = 0.0

	for i, j in allo.grid(L, L, name="smx_row_max"):
		v: Ty = X[i, j]
		if j > i:
			v = -1000000000.0
		if v > M[i]:
			M[i] = v

	for i, j in allo.grid(L, L, name="smx_exp_sum"):
		v: Ty = X[i, j]
		if j > i:
			v = -1000000000.0
		E[i, j] = allo.exp(v - M[i])
		S[i] += E[i, j]

	for i, j in allo.grid(L, L, name="smx_update"):
		Z[i, j] = E[i, j] / S[i]

	return Z


def causal_attention_head[Ty, L, Dh, Mt, Nt](
	Q_h: "Ty[L, Dh]", K_h: "Ty[L, Dh]", V_h: "Ty[L, Dh]"
) -> "Ty[L, Dh]":
	K_t: Ty[Dh, L]
	scores: Ty[L, L] = 0.0
	out_h: Ty[L, Dh] = 0.0

	for i, j in allo.grid(L, Dh, name="k_transpose"):
		K_t[j, i] = K_h[i, j]

	systolic[Ty, Ty, Ty, L, Dh, L, Mt, Nt, "QKT"](Q_h, K_t, scores)

	for i, j in allo.grid(L, L, name="scale_scores"):
		scores[i, j] = scores[i, j] / allo.sqrt(float(Dh))

	probs = gpt_softmax_causal[Ty, L](scores)
	systolic[Ty, Ty, Ty, L, L, Dh, Mt, Nt, "YV"](probs, V_h, out_h)
	return out_h


def gpt2_block[Ty, H, L, D, Dh, Dffn, Mt, Nt](
	X: "Ty[L, D]",
	Wq: "Ty[D, D]",
	bq: "Ty[D]",
	Wk: "Ty[D, D]",
	bk: "Ty[D]",
	Wv: "Ty[D, D]",
	bv: "Ty[D]",
	Wo: "Ty[D, D]",
	bo: "Ty[D]",
	W1: "Ty[D, Dffn]",
	b1: "Ty[Dffn]",
	W2: "Ty[Dffn, D]",
	b2: "Ty[D]",
	ln1_gamma: "Ty[D]",
	ln1_beta: "Ty[D]",
	ln2_gamma: "Ty[D]",
	ln2_beta: "Ty[D]",
) -> "Ty[L, D]":
	Q: Ty[L, D] = 0.0
	K: Ty[L, D] = 0.0
	V: Ty[L, D] = 0.0
	attn_ctx: Ty[L, D]

	systolic[Ty, Ty, Ty, L, D, D, Mt, Nt, "Q"](X, Wq, Q)
	systolic[Ty, Ty, Ty, L, D, D, Mt, Nt, "K"](X, Wk, K)
	systolic[Ty, Ty, Ty, L, D, D, Mt, Nt, "V"](X, Wv, V)

	Q = gpt_bias2d[Ty, L, D](Q, bq)
	K = gpt_bias2d[Ty, L, D](K, bk)
	V = gpt_bias2d[Ty, L, D](V, bv)

	for h in range(H):
		Q_h: Ty[L, Dh]
		K_h: Ty[L, Dh]
		V_h: Ty[L, Dh]
		for i, j in allo.grid(L, Dh, name="mha_split"):
			Q_h[i, j] = Q[i, h * Dh + j]
			K_h[i, j] = K[i, h * Dh + j]
			V_h[i, j] = V[i, h * Dh + j]

		C_h = causal_attention_head[Ty, L, Dh, Mt, Nt](Q_h, K_h, V_h)

		for i, j in allo.grid(L, Dh, name="mha_merge"):
			attn_ctx[i, h * Dh + j] = C_h[i, j]

	O: Ty[L, D] = 0.0
	systolic[Ty, Ty, Ty, L, D, D, Mt, Nt, "O"](attn_ctx, Wo, O)
	O = gpt_bias2d[Ty, L, D](O, bo)

	res1 = gpt_add2d[Ty, L, D](X, O)
	ln1 = gpt_layer_norm2d[Ty, L, D](res1, ln1_gamma, ln1_beta)

	ffn1: Ty[L, Dffn] = 0.0
	systolic[Ty, Ty, Ty, L, D, Dffn, Mt, Nt, "FFN1"](ln1, W1, ffn1)
	ffn1 = gpt_bias2d[Ty, L, Dffn](ffn1, b1)
	ffn1 = gpt_gelu2d[Ty, L, Dffn](ffn1)

	ffn2: Ty[L, D] = 0.0
	systolic[Ty, Ty, Ty, L, Dffn, D, Mt, Nt, "FFN2"](ffn1, W2, ffn2)
	ffn2 = gpt_bias2d[Ty, L, D](ffn2, b2)

	res2 = gpt_add2d[Ty, L, D](ln1, ffn2)
	out = gpt_layer_norm2d[Ty, L, D](res2, ln2_gamma, ln2_beta)
	return out


def gpt2_dataflow[Ty, B, H, L, D, Dh, Dffn, Vocab, Mt, Nt](
	X: "Ty[B, L, D]",
	Wq: "Ty[D, D]",
	bq: "Ty[D]",
	Wk: "Ty[D, D]",
	bk: "Ty[D]",
	Wv: "Ty[D, D]",
	bv: "Ty[D]",
	Wo: "Ty[D, D]",
	bo: "Ty[D]",
	W1: "Ty[D, Dffn]",
	b1: "Ty[Dffn]",
	W2: "Ty[Dffn, D]",
	b2: "Ty[D]",
	ln1_gamma: "Ty[D]",
	ln1_beta: "Ty[D]",
	ln2_gamma: "Ty[D]",
	ln2_beta: "Ty[D]",
	lnf_gamma: "Ty[D]",
	lnf_beta: "Ty[D]",
	Wfc: "Ty[D, Vocab]",
	bfc: "Ty[Vocab]",
) -> "Ty[B, L, Vocab]":
	Y: Ty[B, L, Vocab]

	for b in range(B):
		X_b: Ty[L, D]
		for i, j in allo.grid(L, D, name="copy_batch"):
			X_b[i, j] = X[b, i, j]

		Y_b = gpt2_block[Ty, H, L, D, Dh, Dffn, Mt, Nt](
			X_b,
			Wq,
			bq,
			Wk,
			bk,
			Wv,
			bv,
			Wo,
			bo,
			W1,
			b1,
			W2,
			b2,
			ln1_gamma,
			ln1_beta,
			ln2_gamma,
			ln2_beta,
		)
		Y_b = gpt_layer_norm2d[Ty, L, D](Y_b, lnf_gamma, lnf_beta)

		logits: Ty[L, Vocab] = 0.0
		systolic[Ty, Ty, Ty, L, D, Vocab, Mt, Nt, "FC"](Y_b, Wfc, logits)

		for i, j in allo.grid(L, Vocab, name="add_fc_bias"):
			Y[b, i, j] = logits[i, j] + bfc[j]

	return Y


def make_systolic_sch(m, k, n, mt, nt):
	s = allo.customize(systolic, instantiate=[float32, float32, float32, m, k, n, mt, nt])
	s.partition(s.local_C, dim=0)
	s.partition(s.local_A, dim=1)
	s.partition(s.local_B, dim=2)
	s.pipeline("ak")
	s.pipeline("bk")
	s.pipeline("sj")
	s.unfold("systolic_tile:PE", [0, 1])
	return s


def build_gpt2_dataflow_schedule(B, H, L, D, Dh, Dffn, Vocab, Mt, Nt):
	s = allo.customize(
		gpt2_dataflow,
		instantiate=[float32, B, H, L, D, Dh, Dffn, Vocab, Mt, Nt],
	)

	s.compose(make_systolic_sch(L, D, D, Mt, Nt), id="Q")
	s.compose(make_systolic_sch(L, D, D, Mt, Nt), id="K")
	s.compose(make_systolic_sch(L, D, D, Mt, Nt), id="V")
	s.compose(make_systolic_sch(L, D, D, Mt, Nt), id="O")
	s.compose(make_systolic_sch(L, Dh, L, Mt, Nt), id="QKT")
	s.compose(make_systolic_sch(L, L, Dh, Mt, Nt), id="YV")
	s.compose(make_systolic_sch(L, D, Dffn, Mt, Nt), id="FFN1")
	s.compose(make_systolic_sch(L, Dffn, D, Mt, Nt), id="FFN2")
	s.compose(make_systolic_sch(L, D, Vocab, Mt, Nt), id="FC")

	return s


def export_torch_weights(model):
	blk = model.transformer_blocks[0]
	attn = blk.attention
	ffn = blk.ffn

	weights = {
		"Wq": np.ascontiguousarray(attn.linear_q.weight.detach().cpu().numpy().T.astype(np.float32)),
		"bq": np.ascontiguousarray(attn.linear_q.bias.detach().cpu().numpy().astype(np.float32)),
		"Wk": np.ascontiguousarray(attn.linear_k.weight.detach().cpu().numpy().T.astype(np.float32)),
		"bk": np.ascontiguousarray(attn.linear_k.bias.detach().cpu().numpy().astype(np.float32)),
		"Wv": np.ascontiguousarray(attn.linear_v.weight.detach().cpu().numpy().T.astype(np.float32)),
		"bv": np.ascontiguousarray(attn.linear_v.bias.detach().cpu().numpy().astype(np.float32)),
		"Wo": np.ascontiguousarray(attn.linear_out.weight.detach().cpu().numpy().T.astype(np.float32)),
		"bo": np.ascontiguousarray(attn.linear_out.bias.detach().cpu().numpy().astype(np.float32)),
		"W1": np.ascontiguousarray(ffn.fc1.weight.detach().cpu().numpy().T.astype(np.float32)),
		"b1": np.ascontiguousarray(ffn.fc1.bias.detach().cpu().numpy().astype(np.float32)),
		"W2": np.ascontiguousarray(ffn.fc2.weight.detach().cpu().numpy().T.astype(np.float32)),
		"b2": np.ascontiguousarray(ffn.fc2.bias.detach().cpu().numpy().astype(np.float32)),
		"ln1_gamma": np.ascontiguousarray(blk.norm1.weight.detach().cpu().numpy().astype(np.float32)),
		"ln1_beta": np.ascontiguousarray(blk.norm1.bias.detach().cpu().numpy().astype(np.float32)),
		"ln2_gamma": np.ascontiguousarray(blk.norm2.weight.detach().cpu().numpy().astype(np.float32)),
		"ln2_beta": np.ascontiguousarray(blk.norm2.bias.detach().cpu().numpy().astype(np.float32)),
		"lnf_gamma": np.ascontiguousarray(model.ln_f.weight.detach().cpu().numpy().astype(np.float32)),
		"lnf_beta": np.ascontiguousarray(model.ln_f.bias.detach().cpu().numpy().astype(np.float32)),
		"Wfc": np.ascontiguousarray(model.fc.weight.detach().cpu().numpy().T.astype(np.float32)),
		"bfc": np.ascontiguousarray(model.fc.bias.detach().cpu().numpy().astype(np.float32)),
	}
	return weights


def patch_vitis_kernel_alias_init(project_dir):
	"""Fix generated C++ aliases that Vitis HLS rejects.

	Transforms lines like:
	  float v96[4] = A_fifo[0][0];
	into:
	  float *v96 = A_fifo[0][0];
	"""
	kernel_path = os.path.join(project_dir, "kernel.cpp")
	if not os.path.exists(kernel_path):
		return
	alias_pat = re.compile(
		r"^(\s*)([A-Za-z_]\w*(?:<[^>]+>)?)\s+([A-Za-z_]\w*)\[(\d+)\]\s*=\s*([^;]+);(\s*//.*)?$"
	)
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
		new_lines.append(line)
	if changed:
		with open(kernel_path, "w", encoding="utf-8") as f:
			f.writelines(new_lines)


def main():
	torch.manual_seed(0)
	np.random.seed(0)

	vocab_size = 1
	n_embd = 1024
	n_head = 16
	n_layers = 1
	n_seq = 16
	batch_size = 1

	d_head = n_embd // n_head
	d_ffn = n_embd * 4

	pe_rows = 2
	pe_cols = 2

	model = GPT2(vocab_size, n_embd, n_head, n_layers).eval()
	x_torch = torch.rand(batch_size, n_seq, n_embd)
	golden = model(x_torch).detach().cpu().numpy().astype(np.float32)

	w = export_torch_weights(model)
	x_np = np.ascontiguousarray(x_torch.detach().cpu().numpy().astype(np.float32))
	s = build_gpt2_dataflow_schedule(
		batch_size,
		n_head,
		n_seq,
		n_embd,
		d_head,
		d_ffn,
		vocab_size,
		pe_rows,
		pe_cols,
	)

	def run_allo_module(mod_inst, output=None):
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
			w["W1"],
			w["b1"],
			w["W2"],
			w["b2"],
			w["ln1_gamma"],
			w["ln1_beta"],
			w["ln2_gamma"],
			w["ln2_beta"],
			w["lnf_gamma"],
			w["lnf_beta"],
			w["Wfc"],
			w["bfc"],
		]
		if output is not None:
			args.append(output)
		res = mod_inst(*args)
		return output if output is not None else res

	mod = s.build(target="llvm")
	allo_out = run_allo_module(mod)

	np.testing.assert_allclose(allo_out, golden, atol=2e-2, rtol=2e-2)
	print("Allo GPT2 dataflow output matches PyTorch golden model.")

	run_sw_emu = 0
	run_hw_emu = 1

	if hls.is_available("vitis_hls") and (run_sw_emu or run_hw_emu):
		# Ensure ICD loader can discover OpenCL platforms in this environment.
		if "OCL_ICD_VENDORS" not in os.environ and os.path.isdir("/etc/OpenCL/vendors"):
			os.environ["OCL_ICD_VENDORS"] = "/etc/OpenCL/vendors"
		if run_sw_emu:
			print("Running Vitis HLS sw_emu verification...")
			mod_sw = s.build(
				target="vitis_hls",
				mode="sw_emu",
				project="gpt2_allo_sw_emu",
				configs={"device": "u280"},
			)
			patch_vitis_kernel_alias_init(mod_sw.project)
			out_sw = np.zeros_like(golden, dtype=np.float32)
			run_allo_module(mod_sw, output=out_sw)
			np.testing.assert_allclose(out_sw, golden, atol=2e-2, rtol=2e-2)
			print("Vitis sw_emu output matches PyTorch golden model.")

		if run_hw_emu:
			print("Running Vitis HLS hw_emu verification...")
			mod_hw = s.build(
				target="vitis_hls",
				mode="hw_emu",
				project="gpt2_allo_hw_emu",
				configs={"device": "u280"},
			)
			patch_vitis_kernel_alias_init(mod_hw.project)
			out_hw = np.zeros_like(golden, dtype=np.float32)
			run_allo_module(mod_hw, output=out_hw)
			np.testing.assert_allclose(out_hw, golden, atol=2e-2, rtol=2e-2)
			print("Vitis hw_emu output matches PyTorch golden model.")
	elif run_sw_emu or run_hw_emu:
		print("vitis_hls is not available in this environment, skip sw_emu/hw_emu verification.")


if __name__ == "__main__":
	main()
