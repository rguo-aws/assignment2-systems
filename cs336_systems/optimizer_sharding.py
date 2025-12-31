import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Any, Type, Iterable


class ShardedOptimizer(Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # Will hold only local shard params
        self.params = [p for p in params if p.requires_grad]

        # Call Optimizer superclass constructor
        super().__init__(self.params, kwargs)

        # Construct wrapped optimizer on local shards
        print(f"local params size= {len(self.param_groups)}")
        self.inner_optimizer = optimizer_cls(self.param_groups, **kwargs)

    def _shard_param(self, param: torch.Tensor):
        """Shard a parameter by flattening and chunking."""
        flat = param.detach().view(-1)
        shards = flat.chunk(self.world_size)
        local = shards[self.rank].clone().detach().requires_grad_(True)
        return local

    def add_param_group(self, param_group: dict[str, Any]):
        """Assign parameters to ranks and create local shards."""

        sharded_params = []

        for param in param_group["params"]:
            shard = self._shard_param(param)
            sharded_params.append(shard)

        # Add to wrapped optimizer
        super().add_param_group({"params": sharded_params})

    def step(self, closure=None, **kwargs):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        assert len(self.inner_optimizer.param_groups) == 1
        sharded_params = self.inner_optimizer.param_groups[0]["params"]

        # assign full grad to sharded grad after backward propagation.
        # sharded param is not in the tensor graph

        assert len(sharded_params) == len(self.params)

        for sharded_p, p in zip(sharded_params, self.params):
            sharded_p.grad = p.grad.view(-1).chunk(dist.get_world_size())[self.rank]


        # Update sharded params
        self.inner_optimizer.step(**kwargs)

        # Synchronize updated parameters for next forward propagation
        for p, sp in zip(self.params, sharded_params):
            dist.all_gather(list(p.data.view(-1).chunk(dist.get_world_size())), sp.data)
            p.grad = None

        return loss

