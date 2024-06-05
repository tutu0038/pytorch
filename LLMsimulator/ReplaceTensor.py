import torch

import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import traceback

from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import sys
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

global NET_INITTED
NET_INITTED = True

class Rp(object):
    def __init__(self, raw, replace, **kwargs):
        self.obj = replace
        self.raw = raw

    def __call__(self, *args, **kwargs):
        if not NET_INITTED:
            return self.raw(*args, **kwargs)
        out = self.obj(self.raw, *args, **kwargs)
        return out

def _sigmoid(raw, input):
    return torch.sigmoid(raw, input)

import time
def _bmm(raw, A, B):
    return B
    
from server.profiler_pb2 import ProfilerInput
from server.profiler_grpc import GreeterStub
import asyncio
from grpclib.client import Channel

async def commnicatorasync(r, t, tp) -> None:
    async with Channel('127.0.0.1', 50051) as channel:
        greeter = GreeterStub(channel)
        reply = await greeter.profiler(ProfilerInput(rank = r, time = t, type = tp))
        print(reply.success)

def _Linear(raw ,input ,weight , bias):
    global rank
    dim = input.dim()
    outdim = [0] * dim
    total = 1
    for i in range(dim):
        outdim[i] = input.shape[i]
        total *= outdim[i]
    outdim[dim - 1] = weight.shape[0]
    output = torch.ones(outdim, requires_grad = True)

    time = total * weight.shape[0]
    communicateType = 0
    asyncio.run(commnicatorasync(rank, time, communicateType))

    return output

def _dropout(raw ,input ,p , training, inplace):
    global rank
    asyncio.run(commnicatorasync(rank, time, communicateType))
    communicateType = 2
    return input

from typing import Optional
def _multi_head_attention_forward(raw,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False):

    global _Time
    time = query.shape[0] *  query.shape[1] *  query.shape[2] * 3 / C
    _Time += time
    tp = 2
    asyncio.run(commnicatorasync(rank, time, tp))
    return query, None

def init(rank0):
    global rank
    rank = rank0
    torch.sigmoid = Rp(torch.sigmoid, _sigmoid)
    torch.bmm = Rp(torch.bmm, _bmm)
    nn.functional.linear = Rp(nn.functional.linear, _Linear)
    nn.functional.dropout = Rp(nn.functional.dropout, _dropout)
    nn.functional.multi_head_attention_forward = Rp(nn.functional.multi_head_attention_forward, _multi_head_attention_forward)