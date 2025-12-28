import torch
from torch.nn import Module

import torch.distributed as dist
import torch.multiprocessing as mp


def _grad_hook(handle):
    handle.wait()

class DDPIndividualParameters:
    def __init__(self, module: torch.nn.Module):
        self.model = module
        self.world_size = dist.get_world_size()
        self.handles = []

    def forward(self, *inputs, **kwargs):
        self.model.forward(*inputs, **kwargs)

    def _create_hook(self):
        def hook(param):
            handle = dist.all_reduce(param, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        return hook

    def finish_gradient_synchronization(self):
        grads = [p.grad.data for p in self.model.parameters() if p.grad is not None]

        for grad in grads:
            grad.register_post_accumulate_grad_hook(self._create_hook())

        for handle in self.handles:
            handle.wait()

        for p in grads:
            p.data /= self.world_size
        self.handles.clear()


