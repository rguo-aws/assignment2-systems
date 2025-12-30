from typing import AnyStr, Any

import torch
import dataclasses
from torch.nn import Module

import torch.distributed as dist
import torch.multiprocessing as mp


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super(DDPIndividualParameters, self).__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
            # check to ensure p collects gradient
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._create_hook())

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def _create_hook(self):
        def hook(param):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        return hook

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()

        for p in self.module.parameters():
            if p.requires_grad:
                p.grad.data /= self.world_size

@dataclasses.dataclass
class Bucket:
    size: float
    grads: list[torch.Tensor]
    flat_grad: torch.Tensor = None
    handle: Any = None

class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super(DDPBucketed, self).__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.buckets = []
        self.bucket_size = bucket_size_mb * 1024 * 1024


        for p in reversed(list(self.module.parameters())):
            dist.broadcast(p.data, src=0)
            # check to ensure p collects gradient
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._create_hook())

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def _create_hook(self):
        def hook(param):
            grad_size_mb = param.grad.data.numel() * param.grad.data.element_size()
            if len(self.buckets) == 0:
                self.buckets.append(Bucket(size = grad_size_mb, grads = [param.grad]))
            elif self.buckets[-1].size + grad_size_mb <= self.bucket_size:
                self.buckets[-1].size += grad_size_mb
                self.buckets[-1].grads.append(param.grad)
            else:
                self.buckets[-1].flat_grad = torch._utils._flatten_dense_tensors(self.buckets[-1].grads)
                self.buckets[-1].handle = dist.all_reduce(self.buckets[-1].flat_grad, op=dist.ReduceOp.SUM,
                                                          async_op=True)
                #self.buckets[-1].grads.clear()
                self.buckets.append(Bucket(size = grad_size_mb, grads = [param.grad]))
        return hook

    def finish_gradient_synchronization(self):
        if len(self.buckets) > 0:
            self.buckets[-1].flat_grad = torch._utils._flatten_dense_tensors(self.buckets[-1].grads)
            self.buckets[-1].handle = dist.all_reduce(self.buckets[-1].flat_grad, op=dist.ReduceOp.SUM, async_op=True)

        for bucket in self.buckets:
            bucket.handle.wait()
            for grad, grad_synced in zip(
                    bucket.grads, torch._utils._unflatten_dense_tensors(bucket.flat_grad, bucket.grads)
            ):
                grad.copy_(grad_synced)
                grad.data = grad.data / self.world_size
        self.buckets.clear()
