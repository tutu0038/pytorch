import os

import torch
import LLMprofiler

import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group("cpu:LLMprofiler,cuda:LLMprofiler", rank=0, world_size=1)

x = torch.ones(6)
dist.all_reduce(x)

print(f"cpu allreduce: {x}")

try:
    dist.broadcast(x, 0)
except RuntimeError as e:
    print("got RuntimeError when calling broadcast", e.args)