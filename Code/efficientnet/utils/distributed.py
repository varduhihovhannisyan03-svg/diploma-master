import os

import torch


def setup_for_distributed(is_master):
    """ This function disables printing when not in master process """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    """ Initializing distributed mode """
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    if args.distributed:
        args.local_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.distributed:
        print(f"| distributed init (rank {args.local_rank}): env://", flush=True)
    else:
        print('Warning: Data Parallel is ON. Please use Distributed Data Parallel.')

    setup_for_distributed(args.local_rank == 0)
