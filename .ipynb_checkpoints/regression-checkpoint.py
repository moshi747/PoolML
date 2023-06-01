import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Sigmoid):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def train(X_train, X_test, y_train, y_test, estimator=None, hidden_sizes=[32], lr=1e-2, 
          epochs=50):
    
    df = pd.concat([X_train, y_train], axis=1)
    col = len(df.iloc[0])-1
    train_losses = []
    test_losses = []

    # make core of regression network if it's not there
    if estimator is None:
        reg_net = mlp(sizes=[col]+hidden_sizes+[1])
    else:
        reg_net = estimator
        
    # make batch
    def get_batch(n):
        sample = df.sample(n=n)
        return sample

    # make prediction
    def get_pred(obs):
        return reg_net(obs)*100

    # make loss function
    def compute_loss(pred, true):
        loss = torch.pow(pred - true, 2).mean()
        return loss

    # make optimizer
    optimizer = Adam(reg_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():

        # get predictions
        sample = get_batch(1000)
        pred = get_pred(torch.as_tensor(np.array(sample.iloc[:,:-1]), dtype=torch.float32))

        # take a single regression gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(pred,
                                  torch.as_tensor(np.array(sample.iloc[:,-1:]), dtype=torch.int32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss

    # training loop
    for i in range(epochs):
        batch_loss = train_one_epoch()
        if i%100 == 0:
            train_pred = get_pred(torch.as_tensor(np.array(X_train), dtype=torch.float32))
            train_loss = compute_loss(train_pred, torch.as_tensor(np.array(y_train), dtype=torch.int32)
                                     )
            train_loss = train_loss.detach().numpy()
            train_losses.append(train_loss)
            test_pred = get_pred(torch.as_tensor(np.array(X_test), dtype=torch.float32))
            test_loss = compute_loss(test_pred, torch.as_tensor(np.array(y_test), dtype=torch.int32)
                                    )
            test_loss = test_loss.detach().numpy()
            test_losses.append(test_loss)
    return reg_net, train_losses, test_losses

def test(reg_net, df):
    pred = reg_net(torch.as_tensor(np.array(df.iloc[:,:-1]), dtype=torch.float32))
    true = torch.as_tensor(np.array(df.iloc[:,-1:]), dtype=torch.int32)
    mse = torch.pow(pred*100 - true, 2).mean().item()
    return mse

def pred(reg_net, df):
    pred = reg_net(torch.as_tensor(np.array(df.iloc[:,:-1]), dtype=torch.float32))
    pred_np = pred.detach().numpy()*100
    return np.hstack([np.array(df.iloc[:,-1:]), pred_np])
