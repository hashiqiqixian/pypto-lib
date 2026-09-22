"""Check the unchanged public candidate-selector return contract on A5."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import pypto.language as pl
from golden import TensorSpec, run
from models.deepseek_v4_1_flash.hierarchical_sparse_indexer import hierarchical_sparse_indexer

@pl.jit
def entry(scores: pl.Tensor[[2, 8192], pl.FP32], lengths: pl.Tensor[[2], pl.INT32],
          mask: pl.Out[pl.Tensor[[2, 128], pl.UINT8]]):
    result = hierarchical_sparse_indexer(scores, lengths, mask)
    return result

def golden(tensors):
    tensors["mask"].zero_()
    tensors["mask"][1, :16] = 1

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
torch.set_num_threads(2)
result = run(fn=entry, specs=[
    TensorSpec("scores", [2, 8192], torch.float32, init_value=torch.arange(16384).reshape(2,8192).float()),
    TensorSpec("lengths", [2], torch.int32, init_value=torch.tensor([0, 9], dtype=torch.int32)),
    TensorSpec("mask", [2, 128], torch.uint8, init_value=torch.full((2,128), 37, dtype=torch.uint8)),
], golden_fn=golden, config={"platform": "a5", "device_id": args.device},
    compare_fn={"mask": lambda actual, expected, **kwargs: (torch.equal(actual, expected), "exact public mask")})
if not result.passed:
    raise SystemExit(1)
