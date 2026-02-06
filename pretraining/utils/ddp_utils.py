import os
import torch
import torch.distributed as dist

def ddp_setup(random_seed):
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'     # Enable async error handling
    os.environ['NCCL_TIMEOUT'] = '1800000'                  # Set timeout to 30 minutes (in milliseconds)

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    else:
        raise RuntimeError("CUDA is not available.")

    # set up DDP (distributed data parallel)
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 
    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        device_type = 'cuda'
        torch.cuda.set_device(device)
        print(f"local rank: {ddp_local_rank}, global rank: {ddp_rank}")
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc. (master GPU)
        dist.init_process_group(backend='nccl', rank=ddp_rank, world_size=ddp_world_size)
    else:
        raise ValueError("DDP is not enabled.")     

    if master_process:
        print(f'[MASTER]: DDP established. World size: {ddp_world_size}')

    # sets matmul precisoin to TensorFloat32 (Nvidia's optimized data type) if available or to default
    # from https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision('high')

    return {
        "device": device,
        "device_type": device_type,
        "ddp_world_size": ddp_world_size,
        "ddp_rank": ddp_rank,
        "ddp_local_rank": ddp_local_rank,
        "master_process": master_process,
    }

def ddp_cleanup():
    dist.destroy_process_group()