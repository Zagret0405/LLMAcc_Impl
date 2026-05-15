import importlib.util
import math
import pathlib
import sys
import types

import numpy as np
import torch
import torch.nn as nn


def _load_module(module_name: str, file_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _install_checkpoint_stub():
    # ref_model.py imports gpt_oss.torch.weights.Checkpoint at module import time.
    # The equivalence test below does not use checkpoints, so a small stub is enough.
    if "gpt_oss.torch.weights" in sys.modules:
        return

    gpt_oss = types.ModuleType("gpt_oss")
    gpt_oss_torch = types.ModuleType("gpt_oss.torch")
    gpt_oss_weights = types.ModuleType("gpt_oss.torch.weights")

    class _DummyCheckpoint:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("Checkpoint loading is not used in this equivalence script")

    gpt_oss_weights.Checkpoint = _DummyCheckpoint

    sys.modules["gpt_oss"] = gpt_oss
    sys.modules["gpt_oss.torch"] = gpt_oss_torch
    sys.modules["gpt_oss.torch.weights"] = gpt_oss_weights


class TinyRefOpsGolden(nn.Module):
    def __init__(self, ref_ops, d: int, hq: int, hkv: int, dffn: int, vocab: int, sliding_window: int):
        super().__init__()
        assert hq % hkv == 0
        self.ref_ops = ref_ops
        self.hq = hq
        self.hkv = hkv
        self.q_mult = hq // hkv
        self.dh = d // hq
        self.sliding_window = sliding_window

        self.norm1 = ref_ops.RMSNorm(d)
        self.norm2 = ref_ops.RMSNorm(d)
        self.normf = ref_ops.RMSNorm(d)

        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, hkv * self.dh)
        self.wv = nn.Linear(d, hkv * self.dh)
        self.wo = nn.Linear(d, d)
        self.wg = nn.Linear(d, dffn, bias=False)
        self.wl = nn.Linear(d, dffn, bias=False)
        self.w2 = nn.Linear(dffn, d, bias=False)
        self.fc = nn.Linear(d, vocab, bias=False)

        self.sinks = nn.Parameter(torch.zeros(hq, dtype=torch.float32))
        self.sm_scale = 1.0 / math.sqrt(self.dh)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x shape: [L, ..., Dh]
        L = x.shape[0]
        x3 = x.reshape(L, -1, self.dh)
        y3 = self.ref_ops._apply_rotary_emb(x3, cos, sin)
        return y3.reshape(x.shape)

    def _swiglu_from_ref(self, xg: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        interleaved = torch.empty(
            xg.shape[0],
            xg.shape[1] * 2,
            dtype=xg.dtype,
            device=xg.device,
        )
        interleaved[:, 0::2] = xg
        interleaved[:, 1::2] = xl
        return self.ref_ops.swiglu(interleaved)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x shape: [B, L, D], this golden path uses B=1 for direct parity to ref sdpa signature.
        assert x.shape[0] == 1, "This verifier currently expects B=1"
        xt = x[0]  # [L, D]

        n1 = self.norm1(xt)
        q = self.wq(n1).view(xt.shape[0], self.hkv, self.q_mult, self.dh)
        k = self.wk(n1).view(xt.shape[0], self.hkv, self.dh)
        v = self.wv(n1).view(xt.shape[0], self.hkv, self.dh)

        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        ctx = self.ref_ops.sdpa(
            q,
            k,
            v,
            self.sinks,
            self.sm_scale,
            sliding_window=self.sliding_window,
        )

        o = self.wo(ctx)
        res1 = xt + o

        n2 = self.norm2(res1)
        xg = self.wg(n2)
        xl = self.wl(n2)
        hidden = self._swiglu_from_ref(xg, xl)
        out = res1 + self.w2(hidden)

        out = self.normf(out)
        logits = self.fc(out)
        return logits.unsqueeze(0)


def _copy_tiny_weights_into_golden(tiny: nn.Module, golden: nn.Module):
    with torch.no_grad():
        golden.norm1.scale.copy_(tiny.norm1.scale)
        golden.norm2.scale.copy_(tiny.norm2.scale)
        golden.normf.scale.copy_(tiny.normf.scale)

        golden.wq.weight.copy_(tiny.wq.weight)
        golden.wq.bias.copy_(tiny.wq.bias)
        golden.wk.weight.copy_(tiny.wk.weight)
        golden.wk.bias.copy_(tiny.wk.bias)
        golden.wv.weight.copy_(tiny.wv.weight)
        golden.wv.bias.copy_(tiny.wv.bias)
        golden.wo.weight.copy_(tiny.wo.weight)
        golden.wo.bias.copy_(tiny.wo.bias)

        golden.wg.weight.copy_(tiny.wg.weight)
        golden.wl.weight.copy_(tiny.wl.weight)
        golden.w2.weight.copy_(tiny.w2.weight)
        golden.fc.weight.copy_(tiny.fc.weight)

        golden.sinks.copy_(tiny.sinks)


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
    HQ = 8
    HKV = 2
    Dffn = 96
    Vocab = 32
    Win = 4

    tiny = tiny_mod.TinyGptOssRef(D, HQ, HKV, Dffn, Vocab, Win).eval()
    golden = TinyRefOpsGolden(ref_ops, D, HQ, HKV, Dffn, Vocab, Win).eval()
    _copy_tiny_weights_into_golden(tiny, golden)

    x = torch.randn(B, L, D, dtype=torch.float32)
    cos, sin = tiny_mod.build_rope_cache(L, D // HQ)

    with torch.no_grad():
        y_tiny = tiny(x, cos, sin).float()
        y_golden = golden(x, cos, sin).float()

    max_abs = (y_tiny - y_golden).abs().max().item()
    mean_abs = (y_tiny - y_golden).abs().mean().item()

    print(f"max_abs_diff={max_abs:.8f}")
    print(f"mean_abs_diff={mean_abs:.8f}")

    torch.testing.assert_close(y_tiny, y_golden, rtol=1e-5, atol=1e-5)
    print("PASS: TinyGptOssRef is numerically equivalent to the ref_model-ops golden path.")


if __name__ == "__main__":
    main()
