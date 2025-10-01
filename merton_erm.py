import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import math
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import pickle

torch.set_default_dtype(torch.double)

# hyperparameters (both model and data)
# T = 2
d = 10
k = d 
input_size = 2+d # wealth is one-dimensional, and add one term for bias, and one for innovations
hidden_sizes = [10, 100, 1000]
test_size = 10000
ns = [10, 100, 1000]
p = 16
start_rate = 1 # start_rate of 1 for standard ERM
num_epochs = 10000
trials = 20
x0 = 0
mu = 0.18
vol = 0.44
lmbda = 1
beta = 100
sigma = beta
bins = dict.fromkeys([f"{n}" for n in ns])
for key in bins:
    bins[key] = pd.DataFrame(columns = [f"{r}" for r in hidden_sizes])

# random initialisations
eta = np.random.rand(d)
eta = eta/np.linalg.norm(eta)

# generating out-of-sample data
xi = np.random.normal(loc = mu, scale = np.sqrt(vol), size = test_size)
Z1_test = np.random.uniform(low = -2, high = 2, size = (d, test_size)).T
Z2_test = np.array([i * eta for i in xi])
test_data = np.array(list(zip(Z1_test, Z2_test)))
test_data = torch.from_numpy(test_data)#.float()

# neural network object
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)  
        self.hidden_layer.bias.data.zero_()
        self.hidden_layer.bias.requires_grad = False
        self.sigmoid = nn.Sigmoid()                          
        self.output_layer = nn.Linear(hidden_dim, output_dim)  
        self.output_layer.bias.data.zero_()
        self.output_layer.bias.requires_grad = False
        self.hidden_dim = hidden_dim    
        # self._initialise_weights()                     

    def forward(self, x):
        hidden_activations = self.sigmoid(self.hidden_layer(x))
        raw_output = self.output_layer(hidden_activations)
        # Scale the output by dividing by the number of hidden neurons
        scaled_output = raw_output / self.hidden_dim
        
        return scaled_output#.float()

def regularising(th1, th2, p, r):
    full_th = torch.cat((th1, th2))
    return torch.norm(full_th)**p#/r

def single_loss(X2, lmbda):
    return -(1-torch.exp(-lmbda * X2))

def loss(X2, theta_1, theta_2, lmbda, beta):
    loss_vmap = torch.vmap(single_loss, in_dims = (0, None))
    ells = loss_vmap(X2, lmbda)
    L = torch.mean(ells)*theta_1.shape[0]
    # add on regulariser
    theta_vmap = torch.vmap(regularising, in_dims = (0, 1, None, None))
    ells_by_j = theta_vmap(theta_1, theta_2, p, theta_1.shape[0])
    L_regularising = torch.sum(ells_by_j) / (2*beta**2)
    pseudo_loss = L + L_regularising
    return L, pseudo_loss

def ce(lmbda, v):
    return torch.log(1+v)/lmbda # note we add v as it is the loss, so -1 * utility

num_epochs = 5000
max_epochs = 7000
sigma = 0
start_rate = 1
print("Vanilla ERM")

for trial in range(trials):
    print("Trial: ", trial+1)
    for n in ns:
        print("n: ", n)
        done = False
        while done == False:
            xi = np.random.normal(loc = mu, scale = np.sqrt(vol), size = n)
            Z1 = np.random.uniform(low = -0.5, high = 0.5, size = (d, n)).T
            Z2 = np.array([i * eta for i in xi])
            data = np.array(list(zip(Z1, Z2)))
            data = torch.from_numpy(data)#.float()
            for r in hidden_sizes:
                print("r: ", r)
                model = NeuralNet(input_dim = input_size, hidden_dim = r, output_dim = k)
                learning_rate = start_rate
                optimiser = torch.optim.AdamW(model.parameters(), lr = learning_rate)
                # training loop
                current_loss = 0
                epoch_count = 0
                while epoch_count <= num_epochs or current_loss >= -0.95: 
                    optimiser.zero_grad()
                    X0 = torch.from_numpy(np.zeros(n))#.float()
                    X1 = torch.matmul(torch.from_numpy(np.ones(d)), data[:,0,:].T)/d
                    X1 = X1.view(n, 1) # 1D tensors must be reshaped
                    U1 = model(torch.concat((data[:,1,:], X1, torch.ones(n).view(n, 1)), dim = 1)) 
                    dotter = torch.vmap(torch.dot, in_dims = (0,0))
                    X2 = X1 + dotter(U1, data[:,1,:])
                    actual_loss, pseudo_loss = loss(X2, model.hidden_layer._parameters["weight"],
                                                    model.output_layer._parameters["weight"],
                                                    lmbda, beta)
                    actual_loss.backward()
                    
                    if actual_loss.detach().item() >= current_loss:
                        learning_rate = learning_rate / 2                    
                        current_loss = actual_loss.detach().item()
                    else:
                        current_loss = actual_loss.detach().item()
                    optimiser.step()
                    epoch_count += 1
                    if epoch_count == max_epochs:
                        done = False
                        print("Failed")
                        break
                    else:
                        done = True
                if done == True:
                    with torch.no_grad():
                        X0_test = torch.from_numpy(np.ones(test_size))#.float()
                        X1_test = torch.matmul(torch.from_numpy(np.ones(d)), test_data[:,0,:].T)/d
                        X1_test = X1_test.view(test_size, 1) # 1D tensors must be reshaped
                        U1_test = model(torch.concat((test_data[:,1,:], X1_test, torch.ones(test_size).view(test_size, 1)), dim = 1)) 
                        dotter = torch.vmap(torch.dot, in_dims = (0,0))
                        X2_test = X1_test + dotter(U1_test, test_data[:,1,:])
                        test_loss, pseudo_test_loss = loss(X2_test, model.hidden_layer._parameters["weight"],
                                                model.output_layer._parameters["weight"],
                                                lmbda, beta)
                        bins[f"{n}"].loc[trial, f"{r}"] = torch.abs(actual_loss - test_loss).detach().item()
                elif done == False:
                    break # if broken then we break the for loop and go again


bins.to_pickle("vanilla_diffs.pkl")
