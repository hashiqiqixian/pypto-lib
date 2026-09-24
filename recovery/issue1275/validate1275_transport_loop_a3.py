# External diagnostic: production dispatch stages verbatim except byte dtype aliases
# and omission of the A5-only final tmov_x2zz scale repack. No expert arithmetic.
import os, sys, torch
os.environ['PYPTO_CODEGEN_MAX_WORKERS']='8'
os.environ['PYPTO_COMPILER_TIMEOUT']='120'
from concurrent.futures import ThreadPoolExecutor
from pypto.runtime import device_runner

def bounded_build_pool(*args, **kwargs):
    kwargs['max_workers']=min(8,kwargs.get('max_workers',8))
    print('Binary compiler worker cap:',kwargs['max_workers'],flush=True)
    return ThreadPoolExecutor(*args,**kwargs)

device_runner.ThreadPoolExecutor=bounded_build_pool
from golden import TensorSpec, run
from pypto.ir import DistributedConfig
from models.deepseek_v4_1_flash.ep_transport import combine
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 EP8 transport migrated from V4-Pro and Flash-MTP.

Each function documents its migration source; the data layout follows V4-Pro and the
signal layout is [EP, 1]. Each rank supplies distinct local tokens; dispatch
and combine do not depend on Attention TP ownership.
"""
import pypto.language as pl
import pypto.language.distributed as pld
from models.deepseek_v4_1_flash import config as C

T = C.MOE_TOKENS
D = C.D
TOPK = C.TOPK
N_RANKS = C.EP_SIZE
N_LOCAL = C.N_LOCAL_EXPERTS
N_ROUTES = T * TOPK
RECV_MAX = C.RECV_MAX
MX_GROUP = C.MX_GROUP
K_SCALE = D // MX_GROUP
MAX_PER_SRC = T
AUX_W = 0
AUX_PAD = C.AUX_WIDTH
IDX_PAD = C.ROUTE_WIDTH
SIGNAL_PAD = 1
SCALE_COPY_TILE = 256
SCALE_PACK_TMP = ((64 + K_SCALE + 31) // 32) * 32
RECV_TILE = 16

@pl.jit.inline
def dispatch_bytes(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_mx: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[1, T * K_SCALE], pl.UINT8],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x_out: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.INT8],
    recv_scale_out: pl.Tensor[[1, N_LOCAL * RECV_MAX * K_SCALE], pl.UINT8],
    recv_weight_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
    recv_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
    recv_count_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_meta_local: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, K_SCALE], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    reuse_epoch: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    # Flat 2-D view kept outside the scope so it stays a tensor view, not a tile.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL * RECV_MAX, D])
    # ``quant_mx`` stores MX_A_ZZ bytes physically as
    # [1, M/16, G/2, 16, 2].  Dispatch needs one logical token's scales, so
    # read that backing through its physical ND view instead of scalar-reading
    # an MX-layout tensor (which is intentionally unsupported).
    x_norm_mx_raw = pl.create_tensor([T, D], dtype=pl.INT8)
    with pl.spmd(T, name_hint="dispatch_fp8_raw_copy"):
        copy_row = pl.tile.get_block_idx()
        raw_row = pl.load(x_norm_mx, [copy_row, 0], [1, D])
        raw_row_i8 = raw_row
        x_norm_mx_raw = pl.store(
            raw_row_i8,
            [copy_row, 0],
            x_norm_mx_raw,
        )

    x_norm_scale_raw = pl.create_tensor([1, T * K_SCALE], dtype=pl.UINT8)
    with pl.spmd((T * K_SCALE) // SCALE_COPY_TILE, name_hint="dispatch_e8m0_raw_copy"):
        scale_copy_offset = pl.tile.get_block_idx() * SCALE_COPY_TILE
        raw_scale = pl.load(x_norm_scale, [0, scale_copy_offset], [1, SCALE_COPY_TILE])
        raw_scale_u8 = raw_scale
        x_norm_scale_raw = pl.store(raw_scale_u8, [0, scale_copy_offset], x_norm_scale_raw)
    x_norm_scale_physical = pl.tensor.view(
        x_norm_scale_raw,
        [1, T // 16, K_SCALE // 2, 16, 2],
        layout=pl.ND,
    )

    # Before overwriting any transport window, every rank must have consumed
    # the preceding combine. This includes our own reduction: source order
    # alone does not order a later dispatch against an earlier window reader.
    # Standalone dispatch passes reuse_epoch=0 because it has no combine phase.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reuse_wait",
               allow_early_resolve=False) as _reuse_tid:
        _indices_anchor = pl.read(indices, [0, 0])
        if reuse_epoch > 0:
            for src in pl.range(N_RANKS):
                pld.system.wait(combine_arrived, offsets=[src, 0],
                                expected=reuse_epoch * (N_LOCAL + 1), cmp=pld.WaitCmp.Ge)

    # Meta and payload arrivals ride two independent windows (`arrived` /
    # `data_arrived`). Each producer publishes its current epoch into a unique
    # padded slot, so metadata can gate route construction without waiting for
    # the bulk payload barrier or contending on a shared counter.

    # Stage meta and payload rows locally so their remote publications can use
    # self-draining tensor puts before the matching notifications are issued.
    aux_src = pl.create_tensor([N_ROUTES, AUX_PAD], dtype=pl.FP32)
    route_src = pl.create_tensor([N_ROUTES, IDX_PAD], dtype=pl.INT32)
    scale_src = pl.create_tensor([T, K_SCALE], dtype=pl.UINT8, manual_dep=True)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_stage", deps=[_reuse_tid]) as _stage_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                r = t * TOPK + k
                aux_tile = pl.tile.full([1, AUX_PAD], dtype=pl.FP32, value=0.0)
                aux_weight = pl.read(weights, [t, k])
                pl.tile.write(aux_tile, [0, AUX_W], aux_weight)
                pl.store(aux_tile, [r, 0], aux_src)

                route_tile = pl.tile.full([1, IDX_PAD], dtype=pl.INT32, value=0)
                route_index = pl.cast(r, pl.INT32)
                pl.tile.write(route_tile, [0, 0], route_index)
                pl.store(route_tile, [r, 0], route_src)
            for group in pl.range(K_SCALE):
                scale = pl.read(
                    x_norm_scale_physical,
                    [0, t // 16, group // 2, t % 16, group % 2],
                )
                pl.write(scale_src, [t, group], scale)

    # Phase 1: count routes, publish counts, barrier on meta only, then cumsum ->
    # recv_count_out. Earliest recv_count_out can be produced -- it needs every
    # source's counts but none of the bulk payload.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="dispatch_meta",
        deps=[_reuse_tid],
    ) as _meta_tid:
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        # Count how many routes land in each (dst, loc_e) lane (no payload move).
        cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for d in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                cursor[d * N_LOCAL + e] = 0
        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                dst = eid // N_LOCAL
                loc_e = eid - dst * N_LOCAL
                cursor[dst * N_LOCAL + loc_e] = cursor[dst * N_LOCAL + loc_e] + 1

        # Publish one complete metadata tile per destination: metadata is a tile
        # remote_store, not a sequence of scalar puts, so each destination observes
        # one coherent row.
        meta_tile = pl.tile.full([1, N_LOCAL], dtype=pl.INT32, value=0)
        for dst in pl.range(N_RANKS):
            for e in pl.range(N_LOCAL):
                pl.tile.write(meta_tile, [0, e], cursor[dst * N_LOCAL + e])
            pld.tile.remote_store(
                meta_tile, target=recv_meta, peer=dst, offsets=[my_rank, 0]
            )
            if dst != my_rank:
                pld.system.notify(
                    target=arrived, peer=dst, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

        # Wait for every source's metadata publication.
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=arrived, offsets=[src, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge,
                )

        # Cumsum the published per-source counts into compact local lanes.
        for e in pl.range(N_LOCAL):
            acc = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(recv_meta, [src, e])
                pl.write(recv_meta_local, [src, e], count)
                acc = acc + count
            pl.write(recv_count_out, [e, 0], acc)

    # Phase 2: move the bulk payload (x / aux / route) to each destination lane.
    # Rides its own `data_arrived` window, so it needs no ordering against the meta
    # phase and overlaps it freely.
    # Split over LOCAL EXPERT INDEX (N_LOCAL blocks): block loc_e handles expert
    # loc_e on EVERY destination rank, so the blocking cross-rank puts fan out
    # across N_LOCAL cores. One slot counter per destination rank; token-major
    # order matches the meta pass's per-(dst, loc_e) cumulative count, so the
    # padded lane layout the gather compacts is identical to the single-block push.
    with pl.spmd(N_LOCAL, name_hint="dispatch_push", deps=[_reuse_tid, _stage_tid]) as _push_tid:
        loc_e = pl.tile.get_block_idx()
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        slot_ctr = pl.array.create(N_RANKS, pl.INT32)
        for d in pl.range(N_RANKS):
            slot_ctr[d] = 0
        e_lane_base = loc_e * RECV_MAX + my_rank * MAX_PER_SRC

        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                dst = eid // N_LOCAL
                le = eid - dst * N_LOCAL
                if le == loc_e:
                    slot = slot_ctr[dst]
                    slot_ctr[dst] = slot + 1
                    # lane (loc_e, my_rank, slot) on peer=dst
                    row = e_lane_base + slot
                    r_route = t * TOPK + k
                    pld.tensor.put(
                        dst=recv_x, peer=dst, src=x_norm_mx_raw,
                        dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, D],
                    )
                    pld.tensor.put(
                        dst=recv_scale, peer=dst, src=scale_src,
                        dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, K_SCALE],
                    )
                    pld.tensor.put(
                        dst=recv_weights, peer=dst, src=aux_src,
                        dst_offsets=[row, 0], src_offsets=[r_route, 0], shape=[1, AUX_PAD],
                    )
                    pld.tensor.put(
                        dst=recv_routes, peer=dst, src=route_src,
                        dst_offsets=[row, 0], src_offsets=[r_route, 0], shape=[1, IDX_PAD],
                    )

        # Publish this block's epoch only after its self-draining payload puts.
        # One cache-line-padded slot per source/block avoids shared-word and
        # false-sharing races between the N_LOCAL producers.
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=data_arrived, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    # Each wait block covers the matching producer slot from every remote rank.
    # The whole-grid TaskId then gates gather without serializing 224 waits on
    # one core or allowing a waiting first wave to starve unscheduled producers.
    with pl.spmd(N_LOCAL, name_hint="dispatch_wait", deps=[_meta_tid, _push_tid]) as _wait_tid:
        loc_e = pl.tile.get_block_idx()
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=data_arrived, offsets=[src, 0],
                    expected=moe_epoch * N_LOCAL, cmp=pld.WaitCmp.Ge,
                )

    # Gather lanes into the compact per-expert buffers: one SPMD block per local
    # expert. _wait_tid gates incoming payloads and _push_tid gates this rank's
    # self-peer writes, which are not covered by the remote arrival counters.
    recv_scale_nd = pl.reshape(recv_scale_out, [N_LOCAL * RECV_MAX, K_SCALE])
    with pl.spmd(N_LOCAL, name_hint="dispatch_gather", deps=[_wait_tid, _push_tid]) as _gather_tid:
        e = pl.tile.get_block_idx()
        e_base_row = e * RECV_MAX
        b = pl.cast(0, pl.INDEX)
        for src in pl.range(N_RANKS):
            n = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
            src_base_row = e_base_row + src * MAX_PER_SRC
            for slot in pl.range(n):
                in_row = src_base_row + slot
                out_col = b + slot
                out_row = e_base_row + out_col
                recv_x_raw = pl.load(recv_x, [in_row, 0], [1, D])
                recv_x_mx = recv_x_raw
                recv_x_out_flat = pl.store(recv_x_mx, [out_row, 0], recv_x_out_flat)
                recv_scale_raw = pl.load(recv_scale, [in_row, 0], [1, K_SCALE])
                recv_scale_mx = recv_scale_raw
                recv_scale_nd = pl.store(recv_scale_mx, [out_row, 0], recv_scale_nd)
                pl.write(recv_weight_out, [e, out_col], pl.read(recv_weights, [in_row, AUX_W]))
                pl.write(recv_route_out, [e, out_col], pl.read(recv_routes, [in_row, 0]))
            b = b + n

    return recv_x_out, recv_weight_out, recv_route_out, recv_count_out, recv_meta_local

@pl.jit.inline
def rank_transport(
    indices: pl.Tensor[[T,TOPK],pl.INT32],
    x: pl.Tensor[[T,D],pl.INT8],
    scales: pl.Tensor[[1,T*K_SCALE],pl.UINT8],
    weights: pl.Tensor[[T,TOPK],pl.FP32],
    count: pl.Tensor[[1],pl.INT32],
    rx: pl.Out[pl.Tensor[[N_LOCAL,RECV_MAX,D],pl.INT8]],
    rw: pl.Out[pl.Tensor[[N_LOCAL,RECV_MAX],pl.FP32]],
    rr: pl.Out[pl.Tensor[[N_LOCAL,RECV_MAX],pl.INT32]],
    rc: pl.Out[pl.Tensor[[N_LOCAL,1],pl.INT32]],
    meta: pl.Out[pl.Tensor[[N_RANKS,N_LOCAL],pl.INT32]],
    output: pl.Out[pl.Tensor[[T,D],pl.BF16]],
    scale_out: pl.Out[pl.Tensor[[1,N_LOCAL*RECV_MAX*K_SCALE],pl.UINT8]],
    recv_meta: pld.DistributedTensor[[N_RANKS,N_LOCAL],pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL*RECV_MAX,D],pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL*RECV_MAX,K_SCALE],pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL*RECV_MAX,AUX_PAD],pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL*RECV_MAX,IDX_PAD],pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS,1],pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS,1],pl.INT32],
    routed_output: pld.DistributedTensor[[N_ROUTES,D],pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS,1],pl.INT32],
    rank: pl.Scalar[pl.INT32], epoch: pl.Scalar[pl.INT32],
):
    n=pl.read(count,[0])
    rs=scale_out
    dispatch_bytes(indices,x,scales,weights,rx,rs,rw,rr,rc,meta,
                   recv_meta,recv_x,recv_scale,recv_weights,recv_routes,
                   arrived,data_arrived,combine_arrived,epoch-1,n,rank,epoch)
    y=pl.create_tensor([N_LOCAL,RECV_MAX,D],dtype=pl.BF16)
    yf=pl.reshape(y,[N_LOCAL*RECV_MAX,D])
    with pl.spmd(N_LOCAL,name_hint="diagnostic_weighted_identity"):
        e=pl.tile.get_block_idx()
        rows=pl.read(rc,[e,0])
        for t in pl.range(rows):
            w=pl.read(rw,[e,t])
            v=pl.tile.full([1,D],dtype=pl.FP32,value=1.0)
            v=pl.mul(v,w)
            yf=pl.store(pl.cast(v,pl.BF16),[e*RECV_MAX+t,0],yf)
    shared=pl.create_tensor([T,D],dtype=pl.BF16)
    with pl.spmd(T,name_hint="diagnostic_shared"):
        t=pl.tile.get_block_idx()
        sv=pl.tile.full([1,D],dtype=pl.BF16,value=0.5)
        shared=pl.store(sv,[t,0],shared)
    combine(y,rr,shared,output,meta,routed_output,combine_arrived,n,rank,epoch)
    return rx,rw,rr,rc,meta,output,scale_out

ROUNDS=6
@pl.jit
def rank_sequence(
    indices: pl.Tensor[[ROUNDS, T, TOPK], pl.INT32],
    x: pl.Tensor[[ROUNDS, T, D], pl.INT8],
    scales: pl.Tensor[[ROUNDS, 1, T * K_SCALE], pl.UINT8],
    weights: pl.Tensor[[ROUNDS, T, TOPK], pl.FP32],
    count: pl.Tensor[[ROUNDS, 1], pl.INT32],
    rx: pl.Out[pl.Tensor[[ROUNDS, N_LOCAL, RECV_MAX, D], pl.INT8]],
    rw: pl.Out[pl.Tensor[[ROUNDS, N_LOCAL, RECV_MAX], pl.FP32]],
    rr: pl.Out[pl.Tensor[[ROUNDS, N_LOCAL, RECV_MAX], pl.INT32]],
    rc: pl.Out[pl.Tensor[[ROUNDS, N_LOCAL, 1], pl.INT32]],
    meta: pl.Out[pl.Tensor[[ROUNDS, N_RANKS, N_LOCAL], pl.INT32]],
    output: pl.Out[pl.Tensor[[ROUNDS, T, D], pl.BF16]],
    scale_out: pl.Out[pl.Tensor[[ROUNDS, 1, N_LOCAL * RECV_MAX * K_SCALE], pl.UINT8]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, K_SCALE], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    for step in pl.range(ROUNDS):
        rank_transport(indices[step],x[step],scales[step],weights[step],count[step],rx[step],rw[step],rr[step],rc[step],meta[step],output[step],scale_out[step],recv_meta,recv_x,recv_scale,recv_weights,recv_routes,arrived,data_arrived,routed_output,combine_arrived,rank,step+1)
    return rx,rw,rr,rc,meta,output,scale_out

@pl.jit.host
def transport(
    indices: pl.Tensor[[N_RANKS,ROUNDS,T,TOPK],pl.INT32],
    x: pl.Tensor[[N_RANKS,ROUNDS,T,D],pl.INT8],
    scales: pl.Tensor[[N_RANKS,ROUNDS,1,T*K_SCALE],pl.UINT8],
    weights: pl.Tensor[[N_RANKS,ROUNDS,T,TOPK],pl.FP32],
    count: pl.Tensor[[N_RANKS,ROUNDS,1],pl.INT32],
    rx: pl.Out[pl.Tensor[[N_RANKS,ROUNDS,N_LOCAL,RECV_MAX,D],pl.INT8]],
    rw: pl.Out[pl.Tensor[[N_RANKS,ROUNDS,N_LOCAL,RECV_MAX],pl.FP32]],
    rr: pl.Out[pl.Tensor[[N_RANKS,ROUNDS,N_LOCAL,RECV_MAX],pl.INT32]],
    rc: pl.Out[pl.Tensor[[N_RANKS,ROUNDS,N_LOCAL,1],pl.INT32]],
    meta: pl.Out[pl.Tensor[[N_RANKS,ROUNDS,N_RANKS,N_LOCAL],pl.INT32]],
    output: pl.Out[pl.Tensor[[N_RANKS,ROUNDS,T,D],pl.BF16]],
    scale_out: pl.Out[pl.Tensor[[N_RANKS,ROUNDS,1,N_LOCAL*RECV_MAX*K_SCALE],pl.UINT8]],
):
    recv_meta_buf=pld.alloc_window_buffer([N_RANKS,N_LOCAL],dtype=pl.INT32)
    recv_x_buf=pld.alloc_window_buffer([N_LOCAL*RECV_MAX,D],dtype=pl.INT8)
    recv_scale_buf=pld.alloc_window_buffer([N_LOCAL*RECV_MAX,K_SCALE],dtype=pl.UINT8)
    recv_weights_buf=pld.alloc_window_buffer([N_LOCAL*RECV_MAX,AUX_PAD],dtype=pl.FP32)
    recv_routes_buf=pld.alloc_window_buffer([N_LOCAL*RECV_MAX,IDX_PAD],dtype=pl.INT32)
    arrived_buf=pld.alloc_window_buffer([N_RANKS,1],dtype=pl.INT32)
    data_arrived_buf=pld.alloc_window_buffer([N_RANKS,1],dtype=pl.INT32)
    routed_output_buf=pld.alloc_window_buffer([N_ROUTES,D],dtype=pl.BF16)
    combine_arrived_buf=pld.alloc_window_buffer([N_RANKS,1],dtype=pl.INT32)
    for rank in pl.range(N_RANKS):
        recv_meta=pld.window(recv_meta_buf,[N_RANKS,N_LOCAL],dtype=pl.INT32)
        recv_x=pld.window(recv_x_buf,[N_LOCAL*RECV_MAX,D],dtype=pl.INT8)
        recv_scale=pld.window(recv_scale_buf,[N_LOCAL*RECV_MAX,K_SCALE],dtype=pl.UINT8)
        recv_weights=pld.window(recv_weights_buf,[N_LOCAL*RECV_MAX,AUX_PAD],dtype=pl.FP32)
        recv_routes=pld.window(recv_routes_buf,[N_LOCAL*RECV_MAX,IDX_PAD],dtype=pl.INT32)
        arrived=pld.window(arrived_buf,[N_RANKS,1],dtype=pl.INT32)
        data_arrived=pld.window(data_arrived_buf,[N_RANKS,1],dtype=pl.INT32)
        routed_output=pld.window(routed_output_buf,[N_ROUTES,D],dtype=pl.BF16)
        combine_arrived=pld.window(combine_arrived_buf,[N_RANKS,1],dtype=pl.INT32)
        rank_sequence(indices[rank],x[rank],scales[rank],weights[rank],count[rank],rx[rank],rw[rank],rr[rank],rc[rank],meta[rank],output[rank],scale_out[rank],recv_meta,recv_x,recv_scale,recv_weights,recv_routes,arrived,data_arrived,routed_output,combine_arrived,rank,device=rank)

def fixtures():
    torch.set_num_threads(1)
    count=torch.zeros(ROUNDS,N_RANKS,1,dtype=torch.int32)
    for layer,pair in enumerate(((8,8),(1,0),(65,17))):
        for rank in range(N_RANKS):
            n=pair[rank//4]; width=(n+3)//4
            local=min(width,max(0,n-(rank%4)*width))
            for block in range(2):count[layer*2+block,rank,0]=max(0,min(T,local-block*T))
    indices=torch.empty(ROUNDS,N_RANKS,T,TOPK,dtype=torch.int32)
    x=torch.empty(ROUNDS,N_RANKS,T,D,dtype=torch.int8)
    weights=torch.empty(ROUNDS,N_RANKS,T,TOPK)
    for s in range(ROUNDS):
        for r in range(N_RANKS):
            for t in range(T):
                x[s,r,t]=(torch.arange(D)+r*13+t*3+s*7)%31-15
                for k in range(TOPK):
                    dst=(r+k)%N_RANKS
                    e=(t//4+s*3)%N_LOCAL
                    indices[s,r,t,k]=dst*N_LOCAL+e
                    weights[s,r,t,k]=(k+1+r*8+t*64+s*1024)/128
    logical=(torch.arange(ROUNDS*N_RANKS*T*K_SCALE).reshape(ROUNDS,N_RANKS,T,K_SCALE)%17+119).to(torch.uint8)
    scales=logical.reshape(ROUNDS,N_RANKS,T//16,16,K_SCALE//2,2).permute(0,1,2,4,3,5).contiguous().reshape(ROUNDS,N_RANKS,1,T*K_SCALE)
    return dict(indices=indices,x=x,scales=scales,weights=weights,count=count)

EXPECTED={}
MASKS={}
def golden_round_major(v):
    global EXPECTED,MASKS
    shapes={'rx':(ROUNDS,N_RANKS,N_LOCAL,RECV_MAX,D),'rw':(ROUNDS,N_RANKS,N_LOCAL,RECV_MAX),'rr':(ROUNDS,N_RANKS,N_LOCAL,RECV_MAX),'rc':(ROUNDS,N_RANKS,N_LOCAL,1),'meta':(ROUNDS,N_RANKS,N_RANKS,N_LOCAL),'output':(ROUNDS,N_RANKS,T,D)}
    dtypes={'rx':torch.int8,'rw':torch.float32,'rr':torch.int32,'rc':torch.int32,'meta':torch.int32,'output':torch.bfloat16}
    EXPECTED={n:torch.zeros(sh,dtype=dtypes[n]) for n,sh in shapes.items()}
    MASKS={n:torch.zeros(sh[:-1] if n in ('rx','output') else sh,dtype=torch.bool) for n,sh in shapes.items()}
    MASKS['rc'][:]=True; MASKS['meta'][:]=True
    for s in range(ROUNDS):
        for src in range(N_RANKS):
            for t in range(int(v['count'][s,src,0])):
                acc=torch.full((D,),0.5)
                for k in range(TOPK):
                    dst,e=divmod(int(v['indices'][s,src,t,k]),N_LOCAL)
                    slot=int(EXPECTED['rc'][s,dst,e,0]); EXPECTED['rc'][s,dst,e,0]+=1
                    EXPECTED['meta'][s,dst,src,e]+=1
                    EXPECTED['rx'][s,dst,e,slot]=v['x'][s,src,t]
                    EXPECTED['rw'][s,dst,e,slot]=v['weights'][s,src,t,k]
                    EXPECTED['rr'][s,dst,e,slot]=t*TOPK+k
                    for name in ('rx','rw','rr'):MASKS[name][s,dst,e,slot]=True
                    acc+=v['weights'][s,src,t,k].bfloat16().float()
                EXPECTED['output'][s,src,t]=acc.bfloat16()
                MASKS['output'][s,src,t]=True
        routes=int(EXPECTED['meta'][s].sum())
        assert routes==int(v['count'][s].sum())*TOPK
        print('ROUND',s+1,'tokens',int(v['count'][s].sum()),'routes',routes,flush=True)
    logical=v['scales'].reshape(ROUNDS,N_RANKS,T//16,K_SCALE//2,16,2).permute(0,1,2,4,3,5).reshape(ROUNDS,N_RANKS,T,K_SCALE)
    so=torch.zeros(ROUNDS,N_RANKS,N_LOCAL,RECV_MAX,K_SCALE,dtype=torch.uint8)
    sm=torch.zeros_like(so,dtype=torch.bool)
    cursors={}
    for s in range(ROUNDS):
        for src in range(N_RANKS):
            for t in range(int(v['count'][s,src,0])):
                for k in range(TOPK):
                    dst,e=divmod(int(v['indices'][s,src,t,k]),N_LOCAL)
                    key=(s,dst,e); slot=cursors.get(key,0); cursors[key]=slot+1
                    so[s,dst,e,slot]=logical[s,src,t]; sm[s,dst,e,slot]=True
    EXPECTED['scale_out']=so.reshape(ROUNDS,N_RANKS,1,-1)
    MASKS['scale_out']=sm.reshape(ROUNDS,N_RANKS,1,-1)
    for n in EXPECTED:v[n][:]=EXPECTED[n]

def golden(v):
    round_views={n:t.transpose(0,1) for n,t in v.items()}
    golden_round_major(round_views)
    for n in MASKS:MASKS[n]=MASKS[n].transpose(0,1)
    print('Maximum compact expert rows:',int(round_views['rc'].max()),flush=True)


def comparer(name):
    def compare(a,b,**kw):
        mask=MASKS[name]; a=a[mask]; b=b[mask]
        ok=torch.equal(a,b)
        return ok,'' if ok else f'{name}: mismatches={(a!=b).sum().item()} max={(a.float()-b.float()).abs().max().item()}'
    return compare

if __name__=='__main__':
    vals={n:v.transpose(0,1).contiguous() for n,v in fixtures().items()}
    specs=[TensorSpec(n,list(v.shape),v.dtype,init_value=v) for n,v in vals.items()]
    shapes={'rx':([N_RANKS,ROUNDS,N_LOCAL,RECV_MAX,D],torch.int8),'rw':([N_RANKS,ROUNDS,N_LOCAL,RECV_MAX],torch.float32),'rr':([N_RANKS,ROUNDS,N_LOCAL,RECV_MAX],torch.int32),'rc':([N_RANKS,ROUNDS,N_LOCAL,1],torch.int32),'meta':([N_RANKS,ROUNDS,N_RANKS,N_LOCAL],torch.int32),'output':([N_RANKS,ROUNDS,T,D],torch.bfloat16)}
    shapes['scale_out']=([N_RANKS,ROUNDS,1,N_LOCAL*RECV_MAX*K_SCALE],torch.uint8)
    specs += [TensorSpec(n,s,d) for n,(s,d) in shapes.items()]
    ids=[int(x) for x in os.environ.get('TASK_DEVICE',','.join(map(str,range(N_RANKS)))).split(',')]
    result=run(fn=transport,specs=specs,golden_fn=golden,compare_fn={n:comparer(n) for n in shapes},compile_only='--compile-only' in sys.argv,config=dict(platform='a2a3',distributed_config=DistributedConfig(device_ids=ids,num_sub_workers=0)))
    if not result.passed:raise RuntimeError(result.error)
