"""Temporary device validation of the production TP/mHC boundary; no MoE weights."""
import os
import sys
import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from golden import TensorSpec, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_tp import (
    OUTPUT_T_DYN, prefill_tp_output_reduce_scatter, decode_tp_output_reduce_scatter,
)
from models.deepseek_v4_1_flash.tp_ep_layer import (
    pack_layer_shard, unpack_layer_shard, tp_residual_all_gather, SHARD_MAX,
)
from models.deepseek_v4_1_flash.hc_post import mhc_post

EP=C.EP_SIZE; TP=C.TP_SIZE; D=C.D; HC=C.HC_MULT; HD=C.HC_DIM
T=68; S=(T+TP-1)//TP; BLOCK=C.MOE_TOKENS; ROUNDS=3
T_DYN=C.T_DYN
C1A='--c1a' in sys.argv
from models.deepseek_v4_1_flash.decode_c1a_full import c1a_reduce_scatter
PREFILL='--decode' not in sys.argv and not C1A
REDUCE=prefill_tp_output_reduce_scatter if PREFILL else decode_tp_output_reduce_scatter
WINDOW=C.PREFILL_MAX_TOKENS if PREFILL else C.DECODE_MAX_TOKENS

@pl.jit
def rank_boundary(
    partial: pl.Tensor[[T,D],pl.FP32],
    residual: pl.Tensor[[T,HC,D],pl.FP32],
    post: pl.Tensor[[T,HC],pl.FP32],
    mix: pl.Tensor[[T,HC,HC],pl.FP32],
    counts: pl.Tensor[[1],pl.INT32],
    attention_window: pld.DistributedTensor[[WINDOW,D],pl.FP32],
    attention_signal: pld.DistributedTensor[[TP,1],pl.INT32],
    residual_window: pld.DistributedTensor[[SHARD_MAX,HD],pl.FP32],
    residual_signal: pld.DistributedTensor[[TP,1],pl.INT32],
    output: pl.Out[pl.Tensor[[T,HC,D],pl.FP32]],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    count=pl.read(counts,[0])
    tp_rank=rank%TP
    group_base=rank-tp_rank
    width=(count+TP-1)//TP
    first=pl.min(tp_rank*width,count)
    local_count=pl.min(width,count-first)
    attention=pl.create_tensor([S,D],dtype=pl.BF16)
    if C1A:
        produced=pl.create_tensor([T,D],dtype=pl.FP32)
        with pl.spmd(T,name_hint="test_partial_producer") as produced_ready:
            t=pl.tile.get_block_idx()
            for col in pl.range(0,D,512):
                v=pl.load(partial,[t,col],[1,512])
                produced=pl.store(v,[t,col],produced)
        c1a_reduce_scatter(produced,attention_window,attention_signal,attention,
                           group_base,tp_rank,count,epoch,produced_ready)
    else:
        REDUCE(partial,attention_window,attention_signal,attention,group_base,tp_rank,count,epoch)
    shard=pl.create_tensor([S,HC,D],dtype=pl.FP32)
    for offset in pl.range(0,S,BLOCK):
        active=pl.max(0,pl.min(BLOCK,local_count-offset))
        ab=pl.create_tensor([BLOCK,D],dtype=pl.BF16)
        rb=pl.create_tensor([BLOCK,HC,D],dtype=pl.FP32)
        pb=pl.create_tensor([BLOCK,HC],dtype=pl.FP32)
        mb=pl.create_tensor([BLOCK,HC,HC],dtype=pl.FP32)
        pack_layer_shard(attention,residual,post,mix,ab,rb,pb,mb,first,offset,active)
        result=pl.create_tensor([BLOCK,HC,D],dtype=pl.FP32)
        mhc_post(ab,rb,pb,mb,result)
        unpack_layer_shard(result,shard,offset,active)
    tp_residual_all_gather(shard,residual_window,residual_signal,output,group_base,tp_rank,count,epoch)
    return output

@pl.jit.host
def boundary(
    partial: pl.Tensor[[ROUNDS,EP,T,D],pl.FP32],
    residual: pl.Tensor[[ROUNDS,EP,T,HC,D],pl.FP32],
    post: pl.Tensor[[T,HC],pl.FP32],
    mix: pl.Tensor[[T,HC,HC],pl.FP32],
    counts: pl.Tensor[[ROUNDS,EP,1],pl.INT32],
    output: pl.Out[pl.Tensor[[ROUNDS,EP,T,HC,D],pl.FP32]],
):
    aw=pld.alloc_window_buffer([WINDOW,D],dtype=pl.FP32)
    ac=pld.alloc_window_buffer([TP,1],dtype=pl.INT32)
    rw=pld.alloc_window_buffer([SHARD_MAX,HD],dtype=pl.FP32)
    rc=pld.alloc_window_buffer([TP,1],dtype=pl.INT32)
    for step in pl.range(ROUNDS):
        for rank in pl.range(EP):
            attention_window=pld.window(aw,[WINDOW,D],dtype=pl.FP32)
            attention_signal=pld.window(ac,[TP,1],dtype=pl.INT32)
            residual_window=pld.window(rw,[SHARD_MAX,HD],dtype=pl.FP32)
            residual_signal=pld.window(rc,[TP,1],dtype=pl.INT32)
            rank_boundary(partial[step,rank],residual[step,rank],post,mix,counts[step,rank],
                          attention_window,attention_signal,residual_window,residual_signal,
                          output[step,rank],rank,step+1,device=rank)

def golden(tensors):
    expected=torch.full((ROUNDS,EP,T,HC,D),float('nan'))
    for step in range(ROUNDS):
        for rank in range(EP):
            count=int(tensors['counts'][step,rank,0]); base=rank-rank%TP
            reduced=torch.zeros(count,D)
            for peer in range(TP): reduced+=tensors['partial'][step,base+peer,:count]
            value=tensors['residual'][step,rank,:count]+reduced.bfloat16().float().unsqueeze(1)
            expected[step,rank,:count]=value.bfloat16().float()
    tensors['output'][:]=expected

def compare(actual,expected,**kwargs):
    valid=torch.isfinite(expected)
    ok=torch.equal(actual[valid],expected[valid])
    return ok, '' if ok else f'max error {(actual[valid]-expected[valid]).abs().max().item()}'

if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(1275)
    # Exact binary fractions avoid reference reduction-order ambiguity.
    partial=torch.randint(-8,9,(ROUNDS,EP,T,D)).float()/32
    base=torch.randint(-8,9,(ROUNDS,EP//TP,T,HC,D)).float()/32
    residual=base.repeat_interleave(TP,dim=1)
    counts=torch.empty(ROUNDS,EP,1,dtype=torch.int32)
    for step,pair in enumerate(((8,7),(1,0),(65,17))):
        for rank in range(EP):counts[step,rank,0]=pair[(rank//TP)%2]
    values=dict(partial=partial,residual=residual,post=torch.ones(T,HC),
                mix=torch.eye(HC).expand(T,-1,-1).clone(),counts=counts)
    specs=[TensorSpec(k,list(v.shape),v.dtype,init_value=v) for k,v in values.items()]
    specs.append(TensorSpec('output',[ROUNDS,EP,T,HC,D],torch.float32))
    ids=[int(x) for x in os.environ.get('TASK_DEVICE',','.join(map(str,range(EP)))).split(',')]
    result=run(fn=boundary,specs=specs,golden_fn=golden,compare_fn={'output':compare},
               compile_only='--compile-only' in sys.argv,
               config=dict(platform='a2a3',distributed_config=DistributedConfig(device_ids=ids,num_sub_workers=0)))
    if not result.passed:raise RuntimeError(result.error)

