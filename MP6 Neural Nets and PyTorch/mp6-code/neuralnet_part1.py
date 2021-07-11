# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        num_hidden = 128 
        self.hidden = nn.Linear(in_size, num_hidden)
        self.output = nn.Linear(num_hidden, out_size)

    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = F.relu(self.hidden(x))
        x = F.relu(self.output(x))
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        L = self.loss_fn(x, y)
        L.backward()
        return L

# reference: https://www.analyticsvidhya.com/blog/2019/01/guide-pytorch-neural-networks-case-studies/
#           https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """

    lrate = 0.01
    loss_fn = nn.CrossEntropyLoss()
    in_size = len(train_set[0])
    out_size = 2
    
    nn_model_part1 = NeuralNet(lrate, loss_fn, in_size, out_size)
    #print(sum([param.nelement() for param in nn_model_part1.parameters()]))
    
    # Loss and optimizer
    optimizer = optim.SGD(nn_model_part1.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-2)
    
    losses = []
    yhats = np.zeros(len(dev_set))

    #training part 
    cur_iter = 0
    train_iters = int(np.floor(len(train_set)/batch_size))
    
    #data standarization
    train_set = (train_set - train_set.mean(dim=0)) / train_set.std(dim=0)
    nn_model_part1.train()
    while True:
        #order = torch.randperm(train_set.size()[0])
        #train_set = train_set[order]
        #train_labels = train_labels[order]
        for i in range(train_iters):
            inputs = train_set[batch_size*i:batch_size*(i+1)]
            labels = train_labels[batch_size*i:batch_size*(i+1)]
            
            outputs = nn_model_part1.forward(inputs)

            optimizer.zero_grad()
            loss = nn_model_part1.step(outputs, labels)
            optimizer.step()
            losses.append(loss.item())

            cur_iter+=1
            #print('\rcurrent iters [{:3d}/{:3d}] / loss {:.2E}'.format(cur_iter, n_iter, loss.item()), end='')
            
            if cur_iter >= n_iter: break
        if cur_iter >= n_iter: break
    
    #development part   
    dev_iters = int(np.ceil(len(dev_set)/batch_size))
    
    #data standarization
    dev_set = (dev_set - dev_set.mean(dim=0)) / dev_set.std(dim=0)
    nn_model_part1.eval()
    for i in range(dev_iters):
        inputs = dev_set[batch_size*i:batch_size*(i+1)] if i+1 < dev_iters else dev_set[batch_size*i:]
        
        outputs = nn_model_part1(inputs)
        _, pred_tensor = torch.max(outputs, 1)
        
        if i+1 < dev_iters:
            yhats[batch_size*i:batch_size*(i+1)] = pred_tensor.cpu().numpy()
        else:
            yhats[batch_size*i:] = pred_tensor.cpu().numpy()#
    print()        
    return losses, yhats, nn_model_part1