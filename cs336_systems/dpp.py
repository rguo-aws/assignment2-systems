import os
import timeit

import numpy as np
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

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = "cpu"
        backend = "gloo"
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device

def naive_dpp_worker(rank, world_size, model_config, num_steps, run_benchmark = False):
    WARMUP = 5

    device = setup(rank, world_size)
    torch.manual_seed(42)
    model = BasicsTransformerLM(**model_config).to(device)
    optimizer = AdamW(model.parameters())

    assert num_steps > WARMUP, f"warm up step = {WARMUP} so num_steps should be > {WARMUP}"

    total_times = []
    comm_times = []

    for step in range(num_steps):
        full_batch = torch.randint(0, model_config['vocab_size'], (world_size * 4, model_config['context_length']), device=device)
        local_batch = full_batch[rank * 4: (rank + 1) * 4]

        if run_benchmark:
            torch.cuda.synchronize()
            total_start_time = timeit.default_timer()

        optimizer.zero_grad()
        outputs = model(local_batch)
        loss = outputs.sum()
        loss.backward()

        if run_benchmark:
            torch.cuda.synchronize()
            comm_start_time = timeit.default_timer()

        # dpp tp sync
        for p in model.parameters():
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= world_size

        if run_benchmark:
            torch.cuda.synchronize()
            comm_end_time = timeit.default_timer()
            comm_times.append(comm_end_time - comm_start_time)

        optimizer.step()

        if run_benchmark:
            torch.cuda.synchronize()
            step_end_time = timeit.default_timer()
            total_times.append(step_end_time - total_start_time)

    if rank == 0:
        torch.save(model.state_dict(), "model.pt")

        if run_benchmark:
            total_times = total_times[WARMUP:]
            comm_times = comm_times[WARMUP:]
            avg_total_time = np.mean(total_times) * 1000
            avg_comm_time = np.mean(comm_times) * 1000
            comm_proportion = (avg_comm_time / avg_total_time) * 100

            print(f"Average total time per step: {avg_total_time:.2f} ms")
            print(f"Average communication time per step: {avg_comm_time:.2f} ms")
            print(f"Proportion of time spent on communication: {comm_proportion:.2f}%")


def compare_toy_result_main():
    model_config = {
        "vocab_size": 128,
        "d_model": 64,
        "d_ff": 128,
        "num_layers": 3,
        "num_heads": 4,
        "context_length": 128,
        "rope_theta": 10000.0
    }
    # batch = 16
    p1 = training_no_dpp(model_config, 1, 4)
    for pp in p1:
        print(pp.data)

    world_size = 4
    num_steps = 1
    mp.spawn(fn=naive_dpp_worker, args=(world_size, model_config, num_steps), nprocs=world_size, join=True)

    model = BasicsTransformerLM(**model_config)
    state_dict = torch.load("model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    p2 = model.parameters()

    # compare in main
    for non_parallel_model_parameter, ddp_model_parameter in zip(p1, p2):
        # This parameter was initialized as [2, 2], so we expect its value to remain the same
        assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

def naive_dpp_benchmark():
    model_config = {
        "vocab_size": 128,
        "d_model": 64,
        "d_ff": 128,
        "num_layers": 3,
        "num_heads": 4,
        "context_length": 128,
        "rope_theta": 10000.0
    }

    world_size = 4
    num_steps = 10
    mp.spawn(fn=naive_dpp_worker, args=(world_size, model_config, num_steps, True), nprocs=world_size, join=True)


if __name__ == "__main__":
    #compare_toy_result_main()

    naive_dpp_benchmark()
