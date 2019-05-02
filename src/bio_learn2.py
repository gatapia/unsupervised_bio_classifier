import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

def get_data(data_type):
    mat = scipy.io.loadmat('mnist_all.mat')
    X=torch.zeros((0, 28 * 28), dtype=torch.float)
    y=torch.zeros(0, dtype=torch.long)
    for i in range(10): 
        X_i = torch.from_numpy(mat[data_type + str(i)].astype(np.float)).float()
        X = torch.cat((X, X_i))
        y_i = torch.full(size=(len(X_i),), fill_value=i, dtype=torch.long)
        y = torch.cat((y, y_i))
    return X / 255.0, y

def draw_unsupervised_weights(weights, n_cols, n_rows, img_sz, text=None):
    weights = weights.reshape((-1, img_sz, img_sz))
    indexes = np.random.randint(0, len(weights), n_cols*n_rows)
    weights = weights[indexes]
    fig=plt.figure(figsize=(10, 6))    
    HM=np.zeros((img_sz*n_rows,img_sz*n_cols))
    for idx in range(n_cols * n_rows):
        x, y = idx % n_cols, idx // n_cols
        HM[y*img_sz:(y+1)*img_sz,x*img_sz:(x+1)*img_sz]=weights[idx]
    plt.clf()
    nc=np.amax(np.absolute(HM))
    im=plt.imshow(HM, cmap='bwr', vmin=-nc, vmax=nc)
    fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    if text is not None: plt.title(text)
    plt.axis('off')
    fig.canvas.draw()   

def get_unsupervised_weights(X, n_hidden, n_epochs, batch_size, 
        learning_rate=2e-2, precision=1e-30, anti_hebbian_learning_strength=0.4, lebesgue_norm=2.0, rank=2):
    sample_sz = X.shape[1]    
    weights = torch.rand((n_hidden, sample_sz), dtype=torch.float).cuda()    
    for epoch in range(n_epochs):    
        eps = learning_rate * (1 - epoch / n_epochs)        
        shuffled_epoch_data = X[torch.randperm(X.shape[0]),:]
        for i in range(X.shape[0] // batch_size):
            mini_batch = shuffled_epoch_data[i*batch_size:(i+1)*batch_size,:].cuda()            
            mini_batch = torch.transpose(mini_batch, 0, 1)            
            sign = torch.sign(weights)            
            W = sign * torch.abs(weights) ** (lebesgue_norm - 1)        
            tot_input=torch.mm(W, mini_batch)            
            
            y = torch.argsort(tot_input, dim=0)            
            yl = torch.zeros((n_hidden, batch_size), dtype = torch.float).cuda()
            yl[y[n_hidden-1,:], torch.arange(batch_size)] = 1.0
            yl[y[n_hidden-rank], torch.arange(batch_size)] =- anti_hebbian_learning_strength            
                    
            xx = torch.sum(yl * tot_input,1)            
            xx = xx.unsqueeze(1)                    
            xx = xx.repeat(1, sample_sz)                            
            ds = torch.mm(yl, torch.transpose(mini_batch, 0, 1)) - xx * weights            
            
            nc = torch.max(torch.abs(ds))            
            if nc < precision: nc = precision            
            weights += eps*(ds/nc)
    return weights

def run_test(train_X, train_y, test_X, test_y, model, epochs, loss, batch_size=64, lr=1e-3, verbose=0):
    start = time()
    train_ds = TensorDataset(train_X, train_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = TensorDataset(test_X, test_y)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
    optimizer = Adam(model.parameters(), lr=lr)    
    trainer = create_supervised_trainer(model, optimizer, loss, device='cuda')
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(loss)}, device='cuda')
    
    @trainer.on(Events.COMPLETED)
    def log_completed_validation_results(engine):
        evaluator.run(test_dl)
        avg_accuracy = evaluator.state.metrics['accuracy']
        print("Final Accuracy: {:.2f} Took: {:.0f}s".format(avg_accuracy, time() - start))

    trainer.run(train_dl, max_epochs=epochs) 

class SimpleConvNet(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, out_features)

    def forward(self, x):
        x = x.view(len(x), 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class BioClassifier(nn.Module):
    # Wᵤᵢ is the unsupervised pretrained weight matrix of shape: (n_filters, img_sz)
    def __init__(self, Wᵤᵢ, out_features, n=4.5, β=.01):
        super().__init__()
        self.Wᵤᵢ = Wᵤᵢ.transpose(0, 1) # (img_sz, n_filters)
        self.n = n
        self.β = β
        self.Sₐᵤ = nn.Linear(Wᵤᵢ.size(0), out_features, bias=False)
        
    def forward(self, vᵢ):
        Wᵤᵢvᵢ = torch.matmul(vᵢ, self.Wᵤᵢ)
        hᵤ = F.relu(Wᵤᵢvᵢ) ** self.n
        Sₐᵤhᵤ = self.Sₐᵤ(hᵤ)
        cₐ = torch.tanh(self.β * Sₐᵤhᵤ)
        return cₐ

class BioLoss(nn.Module):
    def __init__(self, out_features, m=6):
        super().__init__()
        self.out_features = out_features
        self.m = m

    def forward(self, cₐ, tₐ): 
        tₐ_ohe = torch.eye(self.out_features, dtype=torch.float, device='cuda')[tₐ]
        tₐ_ohe[tₐ_ohe==0] = -1.
        loss = (cₐ - tₐ_ohe).abs() ** self.m
        return loss.sum()
