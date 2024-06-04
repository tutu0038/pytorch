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
    
from server.helloworld_pb2 import LinearInput
from server.helloworld_grpc import GreeterStub
import asyncio
from grpclib.client import Channel

async def Linearasyn(r, t) -> None:
    async with Channel('127.0.0.1', 50051) as channel:
        greeter = GreeterStub(channel)
        reply = await greeter.Linear(LinearInput(rank = r, time = t))
        print(reply.success)

def _Linear(raw ,input ,weight , bias):
    global _Time
    global rank
    _Time += 0
    dim = input.dim()
    outdim = [0] * dim
    total = 1
    for i in range(dim):
        outdim[i] = input.shape[i]
        total *= outdim[i]
    outdim[dim - 1] = weight.shape[0]
    output = torch.ones(outdim, requires_grad = True)
    time = total * weight.shape[0] / C
    _Time += time
    asyncio.run(Linearasyn(rank, time))
    return output

def _dropout(raw ,input ,p , training, inplace):
    global _Time
    _Time += 0
    return input

async def _multi_head_attention_forwardasync(r, t) -> None:
    async with Channel('127.0.0.1', 50051) as channel:
        greeter = GreeterStub(channel)
        reply = await greeter.attention(LinearInput(rank = r, time = t))
        print(reply.success)


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
    asyncio.run(_multi_head_attention_forwardasync(rank, time))
    return query, None

def init(rank0):
    global rank
    rank = rank0
    torch.sigmoid = Rp(torch.sigmoid, _sigmoid)
    torch.bmm = Rp(torch.bmm, _bmm)
    nn.functional.linear = Rp(nn.functional.linear, _Linear)
    #nn.functional.dropout = Rp(nn.functional.dropout, _dropout)
    nn.functional.multi_head_attention_forward = Rp(nn.functional.multi_head_attention_forward, _multi_head_attention_forward)
    #nn.functional.Conv2d = Rp(nn.functional.Conv2d, _Conv2d)