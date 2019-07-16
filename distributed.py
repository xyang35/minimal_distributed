"""
A minimal example of the distributed model training for single node multiple gpus.
Key functions include:
    - init_process_group
    - DistributedDataParallel
    - DistributedSampler
"""

import torch
import torch.nn as nn
import torch.utils.data
import argparse
import numpy as np
import os

# These libraries are needed for distributed training
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed

# NOTE: You can also use distributed training provided by NVIDIA apex (https://github.com/NVIDIA/apex)
try:
    from apex.parallel import DistributedDataParallel as DDP
except:
    pass


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        # Create 'fake' input data
        N = cfg['N']
        D_in = cfg['D_in']
        x1 = np.random.multivariate_normal(np.ones(D_in)*(-2), np.eye(D_in), N)
        x2 = np.random.multivariate_normal(np.ones(D_in)*2, np.eye(D_in), N)
        x = np.concatenate((x1,x2), axis=0)
        y = np.concatenate((np.zeros(N), np.ones(N)), axis=0)

        data = np.concatenate((y.reshape(-1,1), x), axis=1)
        self.data = torch.from_numpy(data).to(torch.float)
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, index):
        d = self.data[index]
        return d[0], d[1:]


def main():
    # Define some configurations here
    cfg = {'N': 1000,          # input data size
           'D_in': 4,          # input dimensionality
           'batch_size': 8,    # batch size
           'epochs': 5,        # the number of training epochs
           'workers': 4,       # the number of workers used for dataloader
           'multiprocessing_distributed': True    # if set False, DataParallel is used
           }

    np.random.seed(777)
    torch.manual_seed(777)
    
    # Check world size (the number of GPUs in single-node multi-gpu case)
    cfg['world_size'] = torch.cuda.device_count() 
    if cfg['multiprocessing_distributed'] and cfg['world_size'] > 1:
        # Use torch.multiprocessign.spawn to launch distributed processes
        mp.spawn(main_worker, nprocs=cfg['world_size'], args=(cfg,))
    else:
        # Use conventional DataParallel
        main_worker(None, cfg)

def main_worker(gpu, cfg):

    if gpu is not None:
        # Now you are in one of the process. You only need to deal with part of the batch.
        # So adjust the configurations accordingly
        cfg['batch_size'] = int(cfg['batch_size'] / cfg['world_size'])
        cfg['workers'] = int(cfg['workers'] / cfg['world_size'])
        cfg['rank'] = gpu
        print ('Using GPU: {} for training'.format(gpu))
    else:
        cfg['rank'] = 0    # only the main process is used for DataParallel
    
    # The very first step is to initial the distributed package by calling init_process_group()
    # More details for the function in (https://pytorch.org/docs/stable/distributed.html)
    if cfg['multiprocessing_distributed']:
        dist.init_process_group(backend = 'nccl',    # 'nccl' is a common choice for backend 
                                init_method = 'tcp://127.0.0.1:8888',    # tcp initialization is used here (as in ImageNet example), 8888 can be set to any free port
                                world_size = cfg['world_size'],
                                rank = cfg['rank'])

    ################# Model, optimizer and loss function ################

    model = torch.nn.Linear(cfg['D_in'], 2)
    if cfg['multiprocessing_distributed']:
        # For multiprocessing distributed training, the device scope should be set manually
        torch.cuda.set_device(cfg['rank'])
        model.cuda(cfg['rank'])

        # Call DistributedDataParallel constructor, it will automatically make all DDP processes share the same initial values
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg['rank']])
        # NOTE: uncomment the following line (comment the line above) if you want to use apex distributed training
#        model = DDP(model, delay_allreduce=True)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    torch.backends.cudnn.benchmark = True

    ################# Dataset and dataloader ################

    my_dataset = MyDataset(cfg)
    if cfg['multiprocessing_distributed']:
        # A DistributedSampler is needed to for each process to load a subset of the original dataset that is exclusive to it
        sampler = torch.utils.data.distributed.DistributedSampler(my_dataset)
    else:
        sampler = None

    my_dataloader = torch.utils.data.DataLoader(
        my_dataset, batch_size=cfg['batch_size'], shuffle=(sampler is None),    # shuffle is disable if sampler is specified
        num_workers=cfg['workers'], pin_memory=True, sampler=sampler)
    
    ################# Training process ################

    model.train()
    iteration = 0
    for epoch in range(cfg['epochs']):
        # Set_epoch is needed for sampler
        if sampler is not None:
            sampler.set_epoch(epoch)

        # Training process are mostly the same as the single-gpu case
        for _, (targets, inputs) in enumerate(my_dataloader):
            inputs = inputs.cuda()
            targets = targets.cuda()
    
            optimizer.zero_grad()
            # Forward pass
            pred = model(inputs)
            loss = criterion(pred, targets.to(torch.long))
        
            # Backward pass
            loss.backward()
            optimizer.step()
            iteration += 1
        
            if (iteration)%20 == 0:
                # Average losses across processes for logging
                if cfg['multiprocessing_distributed']:
                    reduced_loss = reduce_tensor(loss.data, cfg['world_size'])
                else:
                    reduced_loss = loss.data

                # Printing is only performed on the main process
                # Note that if you need to save your model, it should also be perform ONLY on the main process
                if cfg['rank'] == 0:
                    print("Epoch {} / Iter {}, loss = {:.4}".format(epoch, iteration, reduced_loss.item()))


def reduce_tensor(data, world_size):
    rt = data.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

if __name__ == '__main__':
    main()
