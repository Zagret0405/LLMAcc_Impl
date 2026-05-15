import importlib.util
import pathlib
import sys
import types

import allo
import numpy as np
import torch
import torch.nn as nn
from allo.ir.types import float32


def _load_module(module_name: str, file_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _install_checkpoint_stub():
    # ref_model.py imports gpt_oss.torch.weights.Checkpoint at import time.
    if "gpt_oss.torch.weights" in sys.modules:
        return

    gpt_oss = types.ModuleType("gpt_oss")
    gpt_oss_torch = types.ModuleType("gpt_oss.torch")
    gpt_oss_weights = types.ModuleType("gpt_oss.torch.weights")

    class _DummyCheckpoint:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("Checkpoint loading is not used in this verifier")

    gpt_oss_weights.Checkpoint = _DummyCheckpoint

    sys.modules["gpt_oss"] = gpt_oss
    sys.modules["gpt_oss.torch"] = gpt_oss_torch
    sys.modules["gpt_oss.torch.weights"] = gpt_oss_weights


class RefForwardMoEGolden(nn.Module):
    """MoE golden path following ref_model.MLPBlock.forward math."""

    def __init__(self, ref_ops, d: int, dffn: int, num_experts: int, experts_per_token: int):
        super().__init__()
        self.ref_ops = ref_ops
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.swiglu_limit = 7.0

        self.gate = nn.Linear(d, num_experts, bias=False)
        self.mlp1_weight = nn.Parameter(torch.empty(num_experts, dffn * 2, d))
        self.mlp1_bias = nn.Parameter(torch.empty(num_experts, dffn * 2))
        self.mlp2_weight = nn.Parameter(torch.empty(num_experts, d, dffn))
        self.mlp2_bias = nn.Parameter(torch.empty(num_experts, d))

        with torch.no_grad():
            nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.mlp1_weight, mean=0.0, std=0.02)
            nn.init.normal_(self.mlp1_bias, mean=0.0, std=0.02)
            nn.init.normal_(self.mlp2_weight, mean=0.0, std=0.02)
            nn.init.normal_(self.mlp2_bias, mean=0.0, std=0.02)

    def forward(self, n2: torch.Tensor) -> torch.Tensor:
        # n2 shape: [B, L, D]
        g = self.gate(n2)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.softmax(experts.values, dim=-1)
        expert_indices = experts.indices

        # MLP #1
        mlp1_weight = self.mlp1_weight[expert_indices, ...]  # [B, L, K, 2*Dffn, D]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]  # [B, L, K, 2*Dffn]
        t = torch.einsum("blkcd,bld->blkc", mlp1_weight, n2) + mlp1_bias
        t = self.ref_ops.swiglu(t, limit=self.swiglu_limit)

        # MLP #2
        mlp2_weight = self.mlp2_weight[expert_indices, ...]  # [B, L, K, D, Dffn]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]  # [B, L, K, D]
        t = torch.einsum("blkdc,blkc->blkd", mlp2_weight, t)
        t = t + mlp2_bias

        # Weighted sum of experts
        t = torch.einsum("blkd,blk->bld", t, expert_weights)
        return t


def _to_allo_moe_weights(model: RefForwardMoEGolden):
    gate_w = np.ascontiguousarray(model.gate.weight.detach().cpu().numpy().T.astype(np.float32))

    mlp1_w = model.mlp1_weight.detach().cpu().numpy().astype(np.float32)
    mlp1_b = model.mlp1_bias.detach().cpu().numpy().astype(np.float32)

    wg = np.ascontiguousarray(np.transpose(mlp1_w[:, 0::2, :], (0, 2, 1)))
    wl = np.ascontiguousarray(np.transpose(mlp1_w[:, 1::2, :], (0, 2, 1)))
    bg = np.ascontiguousarray(mlp1_b[:, 0::2])
    bl = np.ascontiguousarray(mlp1_b[:, 1::2])

    mlp2_w = model.mlp2_weight.detach().cpu().numpy().astype(np.float32)
    w2 = np.ascontiguousarray(np.transpose(mlp2_w, (0, 2, 1)))
    b2 = np.ascontiguousarray(model.mlp2_bias.detach().cpu().numpy().astype(np.float32))

    return gate_w, wg, wl, w2, bg, bl, b2


def main():
    base_dir = pathlib.Path(__file__).resolve().parent

    _install_checkpoint_stub()
    ref_ops = _load_module("ref_model_ops", base_dir / "ref_model.py")
    tiny_mod = _load_module("tiny_gpt_oss_mod", base_dir / "gpt-oss-allo.py")

    torch.manual_seed(0)
    np.random.seed(0)

    B = 1
    L = 8
    D = 64
    Dffn = 96
    E = 16
    K = 4

    golden_model = RefForwardMoEGolden(
        ref_ops=ref_ops,
        d=D,
        dffn=Dffn,
        num_experts=E,
        experts_per_token=K,
    ).eval()

    n2_t = torch.randn(B, L, D, dtype=torch.float32)

    with torch.no_grad():
        golden = golden_model(n2_t).squeeze(0).cpu().numpy().astype(np.float32)

    gate_w, wg, wl, w2, bg, bl, b2 = _to_allo_moe_weights(golden_model)

    s = allo.customize(tiny_mod.moe_layer_topk_map4, instantiate=[float32, L, D, Dffn, E])
    mod = s.build(target="llvm")

    n2_np = np.ascontiguousarray(n2_t.squeeze(0).cpu().numpy().astype(np.float32))
    allo_out = mod(n2_np, gate_w, wg, wl, w2, bg, bl, b2)

    max_abs = np.max(np.abs(allo_out - golden))
    mean_abs = np.mean(np.abs(allo_out - golden))

    print(f"max_abs_diff={max_abs:.8f}")
    print(f"mean_abs_diff={mean_abs:.8f}")

    np.testing.assert_allclose(allo_out, golden, atol=3e-2, rtol=3e-2)
    print("PASS: moe_layer_topk_map4 with bias matches ref forward MoE math.")


if __name__ == "__main__":
    main()
