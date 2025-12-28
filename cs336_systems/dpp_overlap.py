import torch
from torch.nn import Module

import torch.distributed as dist
import torch.multiprocessing as mp


def _grad_hook(handle):
    handle.wait()

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super(DDPIndividualParameters, self).__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
            if p.grad is not None:
                p.grad.register_post_accumulate_grad_hook(self._create_hook())

    def forward(self, *inputs, **kwargs):
        self.module.forward(*inputs, **kwargs)

    def _create_hook(self):
        def hook(param):
            handle = dist.all_reduce(param, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        return hook

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()

        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.data /= self.world_size


