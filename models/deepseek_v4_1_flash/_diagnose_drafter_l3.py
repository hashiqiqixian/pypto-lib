"""Temporary EP4 three-layer drafter runtime smoke; not part of PR #1541."""

import inspect

import torch

from golden import ScalarSpec, TensorSpec, run
from pypto.ir import DistributedConfig

import config as C
from decode_moe import _route_bias
from dspark_drafter import l3_dspark_drafter
from expert_routed import build_mxfp4_pair_lut


assert (C.TP_SIZE, C.DP_SIZE, C.EP_SIZE) == (1, 4, 4)

DTYPES = {
    C.pl.BF16: torch.bfloat16,
    C.pl.FP32: torch.float32,
    C.pl.FP8E4M3FN: torch.float8_e4m3fn,
    C.pl.FP8E8M0: torch.float8_e8m0fnu,
    C.pl.INT16: torch.int16,
    C.pl.INT32: torch.int32,
    C.pl.INT64: torch.int64,
    C.pl.UINT8: torch.uint8,
}


def extent(dim):
    if isinstance(dim, int):
        return dim
    name = repr(dim)
    for marker, value in (
        ("DSPARK_QUERY_T_DYN", C.DSPARK_QUERY_WIDTH),
        ("ORI_BLOCKS_DYN", 1),
        ("TABLE_DYN", 1),
        ("B_DYN", 1),
        ("T_DYN", C.DECODE_ROWS_PER_REQUEST),
    ):
        if marker in name:
            return value
    raise ValueError(f"No fixture extent for {dim!r}")


signature = inspect.signature(l3_dspark_drafter._func)
specs = []
for name in l3_dspark_drafter.param_names:
    annotation = signature.parameters[name].annotation
    if name in ("attention_epoch", "moe_epoch"):
        specs.append(ScalarSpec(name, torch.int32, 1, compile_runtime=True))
        continue
    shape = [extent(dim) for dim in annotation.shape]
    dtype = DTYPES[annotation.dtype]
    initial = 0
    if name == "context_slots":
        initial = -1
    elif name.endswith("norm_weight") or name in ("context_rope_cos", "query_rope_cos"):
        initial = 1
    elif name == "mxfp4_pair_lut":
        initial = build_mxfp4_pair_lut().unsqueeze(0).expand(C.EP_SIZE, -1, -1).contiguous()
    elif name == "correction_bias":
        initial = _route_bias().reshape(1, 1, -1).expand(C.EP_SIZE, C.DSPARK_DRAFT_LAYERS, -1).contiguous()
    specs.append(TensorSpec(name, shape, dtype, init_value=initial))


def golden(values):
    values["head_hidden"].zero_()


def check_zero(actual, _expected, **_kwargs):
    value = actual.float()
    maximum = value.abs().max().item()
    return bool(torch.isfinite(value).all()) and maximum <= 1e-3, f"max abs={maximum}"


result = run(
    fn=l3_dspark_drafter,
    specs=specs,
    golden_fn=golden,
    compare_fn={
        "head_hidden": check_zero,
        "window_cache": lambda *_args, **_kwargs: (True, "cache numerical contents not checked"),
        "window_cache_scale": lambda *_args, **_kwargs: (True, "cache scale numerical contents not checked"),
    },
    config={
        "platform": "a5",
        "distributed_config": DistributedConfig(device_ids=[0, 1, 2, 3], num_sub_workers=0),
        "ring_heap": (4 * 1024**3,) * C.EP_SIZE,
    },
)
print(result)
if not result.passed:
    raise SystemExit(1)
