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

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        # Create 'fake' input data
        N = cfg['N']
        D_in = cfg['D_in']
        x1 = np.random.multivariate_normal(np.ones(D_in)*(-1), np.eye(D_in), N)
        x2 = np.random.multivariate_normal(np.ones(D_in), np.eye(D_in), N)
        x = np.concatenate((x1,x2), axis=0)
        y = np.concatenate((np.zeros(N), np.ones(N)), axis=0)

        self.data = np.concatenate((y, x), axis=1)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        d = self.data[index]
        return d[0].reshape(1,-1), d[1:].reshape(1,-1)

def main():
    # Define some configurations here
    cfg = {'N': 1000,          # input data size
           'D_in': 4,          # input dimensionality
           'batch_size': 8,    # batch size
           'epochs': 5,        # the number of training epochs
           'workers': 4,       # the number of workers used for dataloader
           }
    
    ################# Model, optimizer and loss function ################

    model = torch.nn.Linear(cfg['D_in'], 2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    torch.backends.cudnn.benchmark = True

    ################# Dataset and dataloader ################

    my_dataset = MyDataset(cfg)
    my_dataloader = torch.utils.data.Dataloader(
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
            loss = criterion(pred, targets)
        
            # Backward pass
            loss.backward()
            optimizer.step()
            iteration += 1
        
            if (iteration+1)%20 == 0:
                print("Epoch {} / Iter {}, loss = {}".format(epoch, iteration, loss.item()))

if __name__ == '__main__':
    main()