"""
A minimal example of the basic model training procedure, including:
    - Dataset / Dataloader
    - Model / Optimizer
    - Training
"""

import torch
import torch.nn as nn
import torch.utils.data
import argparse
import numpy as np
import os
import pdb

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
           }

    np.random.seed(777)
    torch.manual_seed(777)
    
    ################# Model, optimizer and loss function ################

    model = torch.nn.Linear(cfg['D_in'], 2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    torch.backends.cudnn.benchmark = True

    ################# Dataset and dataloader ################

    my_dataset = MyDataset(cfg)
    my_dataloader = torch.utils.data.DataLoader(
        my_dataset, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['workers'], pin_memory=True)
    

    ################# Training process ################

    model.train()
    iteration = 0
    for epoch in range(cfg['epochs']):
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
                print("Epoch {} / Iter {}, loss = {:.4}".format(epoch, iteration, loss.item()))

if __name__ == '__main__':
    main()
