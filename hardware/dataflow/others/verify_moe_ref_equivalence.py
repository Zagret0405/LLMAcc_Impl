import importlib.util
import pathlib
import sys
import types

import numpy as np
import torch

import allo
from allo.ir.types import float32


def _load_module(module_name: str, file_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _install_checkpoint_stub():
    if "gpt_oss.torch.weights" in sys.modules:
        return

    gpt_oss = types.ModuleType("gpt_oss")
    gpt_oss_torch = types.ModuleType("gpt_oss.torch")
    gpt_oss_weights = types.ModuleType("gpt_oss.torch.weights")

    class _DummyCheckpoint:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("Checkpoint loading is not used in this MoE verifier")

    gpt_oss_weights.Checkpoint = _DummyCheckpoint

    sys.modules["gpt_oss"] = gpt_oss
    sys.modules["gpt_oss.torch"] = gpt_oss_torch
    sys.modules["gpt_oss.torch.weights"] = gpt_oss_weights


def _swiglu_from_ref(ref_ops, xg: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
    interleaved = torch.empty(
        *xg.shape[:-1],
        xg.shape[-1] * 2,
        dtype=xg.dtype,
        device=xg.device,
    )
    interleaved[..., 0::2] = xg
    interleaved[..., 1::2] = xl
    return ref_ops.swiglu(interleaved)


def moe_golden_from_ref_ops(ref_ops, tiny, n2: torch.Tensor) -> torch.Tensor:
    logits = torch.einsum("bld,de->ble", n2, tiny.wgate.weight.T)
    topk = min(4, tiny.experts_per_token, tiny.num_experts)
    experts = torch.topk(logits, k=topk, dim=-1, sorted=True)

    expert_weights = torch.softmax(experts.values, dim=-1)
    expert_indices = experts.indices

    if topk < 4:
        pad_w = torch.zeros(
            n2.shape[0], n2.shape[1], 4 - topk, device=n2.device, dtype=n2.dtype
        )
        pad_i = torch.zeros(
            n2.shape[0],
            n2.shape[1],
            4 - topk,
            device=n2.device,
            dtype=expert_indices.dtype,
        )
        expert_weights = torch.cat([expert_weights, pad_w], dim=-1)
        expert_indices = torch.cat([expert_indices, pad_i], dim=-1)

    wg_sel = tiny.wg_experts[expert_indices]
    bg_sel = tiny.bg_experts[expert_indices]
    wl_sel = tiny.wl_experts[expert_indices]
    bl_sel = tiny.bl_experts[expert_indices]
    w2_sel = tiny.w2_experts[expert_indices]
    b2_sel = tiny.b2_experts[expert_indices]

    xg = torch.einsum("blkdf,bld->blkf", wg_sel, n2) + bg_sel
    xl = torch.einsum("blkdf,bld->blkf", wl_sel, n2) + bl_sel

    hidden = _swiglu_from_ref(ref_ops, xg, xl)
    expert_out = torch.einsum("blkfd,blkf->blkd", w2_sel, hidden) + b2_sel
    return torch.sum(expert_out * expert_weights.unsqueeze(-1), dim=2)


def allo_moe_forward(gpt_mod, n2: np.ndarray, weights: dict, L: int, D: int, Dffn: int, E: int, K: int):
    s = allo.customize(
        gpt_mod.moe_layer_topk_map4,
        instantiate=[float32, L, D, Dffn, E, K],
    )
    mod = s.build(target="llvm")
    return mod(
        n2,
        weights["Wgate"],
        weights["Wg_experts"],
        weights["bg_experts"],
        weights["Wl_experts"],
        weights["bl_experts"],
        weights["W2_experts"],
        weights["b2_experts"],
    )


def main():
    base_dir = pathlib.Path(__file__).resolve().parent

    _install_checkpoint_stub()
    ref_ops = _load_module("ref_model_ops", base_dir / "ref_model.py")
    gpt_mod = _load_module("gpt_oss_allo_mod", base_dir / "gpt-oss-allo.py")

    torch.manual_seed(0)
    np.random.seed(0)

    B = 1
    L = 8
    D = 64
    HQ = 8
    HKV = 2
    Dffn = 96
    Vocab = 32
    Win = 4
    E = 32
    K = 4

    tiny = gpt_mod.TinyGptOssRef(D, HQ, HKV, Dffn, Vocab, Win, E, K).eval()
    n2 = torch.randn(B, L, D, dtype=torch.float32)

    with torch.no_grad():
        y_tiny = tiny._moe_forward(n2).float()
        y_golden = moe_golden_from_ref_ops(ref_ops, tiny, n2).float()

    max_abs_tiny = (y_tiny - y_golden).abs().max().item()
    mean_abs_tiny = (y_tiny - y_golden).abs().mean().item()
    print(f"tiny_vs_golden max_abs_diff={max_abs_tiny:.8f}")
    print(f"tiny_vs_golden mean_abs_diff={mean_abs_tiny:.8f}")
    torch.testing.assert_close(y_tiny, y_golden, rtol=1e-5, atol=1e-5)

    weights = gpt_mod.export_ref_weights(tiny)
    n2_np = np.ascontiguousarray(n2[0].detach().cpu().numpy().astype(np.float32))
    y_allo = allo_moe_forward(gpt_mod, n2_np, weights, L, D, Dffn, E, K)

    y_allo_t = torch.from_numpy(np.asarray(y_allo)).float()
    y_golden_l0 = y_golden[0]
    max_abs_allo = (y_allo_t - y_golden_l0).abs().max().item()
    mean_abs_allo = (y_allo_t - y_golden_l0).abs().mean().item()
    print(f"allo_vs_golden max_abs_diff={max_abs_allo:.8f}")
    print(f"allo_vs_golden mean_abs_diff={mean_abs_allo:.8f}")
    torch.testing.assert_close(y_allo_t, y_golden_l0, rtol=1e-5, atol=1e-5)

    print("PASS: MoE module matches the reference-based golden implementation.")


if __name__ == "__main__":
    main()
