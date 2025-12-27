import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

def training_no_dpp(model_config, num_steps, world_size):
    torch.manual_seed(42)
    model = BasicsTransformerLM(**model_config)
    optimizer = AdamW(model.parameters())

    for step in range(num_steps):
        full_batch = torch.randint(0, model_config['vocab_size'], (world_size * 4, model_config['context_length']))
        #local_batch = full_batch[rank * 4: (rank + 1) * 4]
        optimizer.zero_grad()
        outputs = model(full_batch)
        loss = outputs.sum()
        loss.backward()
        optimizer.step()

    return model.parameters()

def setup(rank, world_size, use_cpu: bool = False):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if use_cpu:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def training_dpp_worker(rank, world_size, model_config, num_steps, use_cpu: bool = False):
    setup(rank, world_size, use_cpu)
    torch.manual_seed(42)
    model = BasicsTransformerLM(**model_config)
    optimizer = AdamW(model.parameters())

    for step in range(num_steps):
        full_batch = torch.randint(0, model_config['vocab_size'], (world_size * 4, model_config['context_length']))
        local_batch = full_batch[rank * 4: (rank + 1) * 4]
        optimizer.zero_grad()
        outputs = model(local_batch)
        loss = outputs.sum()
        loss.backward()

        # dpp tp sync
        for p in model.parameters():
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= world_size

        optimizer.step()

    if rank == 0:
        torch.save(model.state_dict(), "model.pt")

def training_dpp(model_config, num_steps, use_cpu):
    world_size = 4
    mp.spawn(fn=training_dpp_worker, args=(world_size, model_config, num_steps, use_cpu), nprocs=world_size, join=True)

if __name__ == "__main__":
    model_config = {
        "vocab_size": 128,
        "d_model": 64,
        "d_ff": 128,
        "num_layers": 3,
        "num_heads": 4,
        "context_length": 128,
        "rope_theta" : 10000.0
    }
    # batch = 16
    p1 = training_no_dpp(model_config, 1, 4)
    for pp in p1:
        print(pp.data)

    training_dpp(model_config, 1, True)

    model = BasicsTransformerLM(**model_config)
    state_dict = torch.load("model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    p2 = model.parameters()

    #compare in main
    for non_parallel_model_parameter, ddp_model_parameter in zip(p1, p2):
        # This parameter was initialized as [2, 2], so we expect its value to remain the same
        assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

