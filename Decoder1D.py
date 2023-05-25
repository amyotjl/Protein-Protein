import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools
import joblib
from matplotlib import pyplot as plot
import os
import gc
from load_dataset import get_dataloader
import utils
import argparse
from ray import tune

class ConvNet1D(nn.Module):
    def __init__(self, input_size, output_size, in_channels,
                 conv_k = 3, conv_s=1, conv_p=1, pool_k=2, pool_s=2,
                 use_batch_norm=False, dropout_rate = 0, activation_function = nn.ReLU()):
        super(ConvNet1D, self).__init__()
        self.layers = nn.ModuleList()
        self.in_channels = in_channels
        self.final_output_dim = output_size
        self.conv_k = conv_k
        self.conv_s = conv_s
        self.conv_p = conv_p
        self.pool_k = pool_k
        self.pool_s = pool_s
        out_dim = input_size
        for i in range(len(self.in_channels)-1):
            self.layers.append(nn.Conv1d(self.in_channels[i], self.in_channels[i+1],  kernel_size=self.conv_k, stride=self.conv_s, padding=self.conv_p))
            self.layers.append(activation_function)
            self.layers.append(nn.MaxPool1d(kernel_size=self.pool_k, stride=self.pool_s))
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(self.in_channels[i+1]))
            self.layers.append(nn.Dropout(p=dropout_rate))
            out_dim = self.compute_dim(out_dim, self.conv_k, self.conv_s, self.conv_p, self.pool_k, self.pool_s)
            
        self.conv_net = nn.Sequential(*self.layers)
        self.fc = nn.Linear(int(out_dim* self.in_channels[-1]), self.final_output_dim[0]*self.final_output_dim[1])
        
    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out= out.view(out.shape[0], *self.final_output_dim)
        return out
    
    def compute_dim(self, dim, conv_k, conv_s, conv_p, pool_k, pool_s):
        dim = np.floor(((dim - conv_k + 2 * conv_p)/conv_s)+1)
        return np.floor(((dim - pool_k)/pool_s)+1)

def train(conv_net, train_loader, test_loader, optimizer:optim, epochs:int = 10, loss_fn = None):
    train_losses = []
    test_losses = []

    for i in range(epochs):
        print('epoch {}'.format(i))
        epoch_loss = []
        test_loss = []

        # Training
        for _, (batch,labels) in tqdm(enumerate(train_loader)):
            conv_net.train()
            batch = batch.to(gpu)
            labels = labels.cpu()
            optimizer.zero_grad()
            output = conv_net(batch)
            output = output.cpu()

            loss_train = loss_fn(output, labels)
            loss_train.backward()
            optimizer.step()
            epoch_loss.append(loss_train.item())

        # Test Set
        with torch.no_grad():    
            for _, (batch,labels) in enumerate(test_loader):
                conv_net.eval()
                batch = batch.to(gpu)
                labels = labels.cpu()
                output = conv_net(batch)
                output = output.cpu()
                loss_test = loss_fn(output, labels)
                test_loss.append(loss_test.item())

        train_losses.append(np.average(epoch_loss))
        test_losses.append(np.average(test_loss))
    return train_losses, test_losses
    
def load_all_data(AA_random_matrices_path, embeddings_path, dataloader_batch_size, train_size):
    random_matrices = joblib.load(AA_random_matrices_path)
    return get_dataloader(embeddings_path, random_matrices, batch_size = dataloader_batch_size, train_size=train_size)

def get_activation_function(name):
    activation_functions = {
        'ReLU': nn.ReLU(),
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh(),
        'LeakyReLU': nn.LeakyReLU(),
        'ELU': nn.ELU(),
        'SELU': nn.SELU(),
        'Softmax': nn.Softmax(),
        'Softplus': nn.Softplus(),
        'Softsign': nn.Softsign(),
        'Hardtanh': nn.Hardtanh(),
        'LogSigmoid': nn.LogSigmoid(),
        'Softshrink': nn.Softshrink(),
        'ReLU6': nn.ReLU6(),
        'RReLU': nn.RReLU(),
        'CELU': nn.CELU(),
        'GLU': nn.GLU()
    }

    return activation_functions.get(name, None)

def get_optimizer(name):
    optimizers = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'Adagrad': optim.Adagrad,
        'Adadelta': optim.Adadelta,
        'RMSprop': optim.RMSprop,
        'SparseAdam': optim.SparseAdam,
        'AdamW': optim.AdamW,
        'Adamax': optim.Adamax,
        'ASGD': optim.ASGD,
        'Rprop': optim.Rprop,
        'LBFGS': optim.LBFGS,
        'Radam': optim.RAdam,
    }

    return optimizers.get(name, None)

def get_loss_function(name):
    loss_functions = {
        'MSELoss': nn.MSELoss(),
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'BCELoss': nn.BCELoss(),
        'KLDivLoss': nn.KLDivLoss(),
        'L1Loss': nn.L1Loss(),
        'SmoothL1Loss': nn.SmoothL1Loss()
    }
    
    return loss_functions.get(name, None)

def run(args):
    device = torch.device('cpu')
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Using GPU')
    else :
        print('Using CPU')
        

    print('Reading random matricies.')
    AA_random_matrices = joblib.load(args.random_matrices)
    print('Done')
    print('Getting Dataloaders')
    train_loader, test_loader = get_dataloader(args.embeddings_path, AA_random_matrices, args.batch_size, train_size=args.training_size)
    print('Done')
    model = ConvNet1D(args.embeddings_length,
                      args.output_shape,
                      args.channels,
                      args.conv_k, args.conv_s, args.conv_p,
                      args.pool_k,args.pool_s,
                      args.batch_normalization,
                      args.dropout_rate,
                      get_activation_function(args.activation_func)
                      ).to(device)
    optimizer = get_optimizer(args.optimizer)(model.parameters(), lr = args.learning_rate)
    loss_function = get_loss_function(args.loss_func).to(device)

    retsults = train(model, train_loader, test_loader, optimizer, epochs=args.epochs, loss_fn=loss_function)
    
config = {
    'channels':[[1,32,64],[1,16,32],[1,4,8,16], [1,16,32,64]],
    'conv_k': range(2,5),
    'conv_s': range(0,2),
    'conv_p': range(0,2),
    'pool_k': range(2,5),
    'pool_s': range(0,2),
    'activation_func': [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(), nn.LogSigmoid()],
    'loss_func': [nn.MSELoss(), nn.CrossEntropyLoss(), nn.L1Loss()],
    'optimizer' : [optim.SGD, optim.Adagrad, optim.AdamW, optim.ASGD],
    'training_size': [0.9, 0.85, 0.8],
    'learning_rate': [0.01,0.001,0.0001],
    'batch_size' : 256,
    'dropout_rate' : [0, 0.15, 0.25],
    'batch_normalization' : [True, False]
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script creates a Conv1D model based on pre-calculated embeddings.')
    parser.add_argument('--embeddings_length', type=int, default=1024)
    parser.add_argument('-outs','--output_shape', type=tuple, help='Shape of the output. Should be longest aminoacid sequence x number of unique aminoacid + 1 (the padding one)', default=(1965,22)) # 
    parser.add_argument('-ch', '--channels', type=list, help='List of channels for the model. The length of the list will be the number of layers. The first number MUST BE the same as the first number of the input shape.', default=[1,4,8,16])
    parser.add_argument('--use_gpu', type=bool, help="Whether or not to use the GPU if available", default = True)
    parser.add_argument( '-embs','--embeddings_path', type=str, help="Path to the embedding file", default = 'data/Prots_embeddings_1d.joblib')
    parser.add_argument('-rm','--random_matrices', type=str, help="Path to the random matrices file", default = 'data/AA_random_matrices.joblib')
    parser.add_argument('--conv_k', type=int, default=3, help='Convolution kernel size')
    parser.add_argument('--conv_s', type=int, default=1, help='Convolution stride')
    parser.add_argument('--conv_p', type=int, default=1, help='Convolution padding')
    parser.add_argument('--pool_k', type=int, default=2, help='Pooling kernel size')
    parser.add_argument('--pool_s', type=int, default=2, help='Pooling stride')
    parser.add_argument('-af','--activation_func', type=str, choices=['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU', 'SELU', 'Softmax', 'Softplus', 'Softsign', 'Hardtanh', 'LogSigmoid', 'Softshrink', 'ReLU6', 'RReLU', 'CELU', 'GLU'], default = 'ReLU')
    parser.add_argument('-opt','--optimizer', type=str, choices=['SGD', 'Adam', 'Adagrad', 'Adadelta', 'RMSprop', 'SparseAdam', 'AdamW', 'Adamax', 'ASGD', 'Rprop', 'LBFGS', 'Lookahead', 'Ranger', 'Radam', 'NovoGrad', 'RangerPlus', 'RangerQH'], default = 'Adam')
    parser.add_argument('-lr','--learning_rate', type=float, default = 0.001)
    parser.add_argument('-bs','--batch_size', type=int, default = 128)
    parser.add_argument('-ts','--training_size', type=float, choices = [i/100 for i in range(101)], help = 'Size of the training set. Test size will be the complement. This value must be between 0 and 1.', default = 0.8)
    parser.add_argument('-e','--epochs', type=int, default = 30)
    parser.add_argument('-dr','--dropout_rate', type=float, help="Default is 0 so that no dropout is applied.", default = 0)
    parser.add_argument('-bn','--batch_normalization', type=bool, default = False)
    parser.add_argument('--loss_func', type=str, default = 'MSELoss', choices=['MSELoss', 'CrossEntropyLoss', 'BCELoss', 'KLDivLoss', 'L1Loss', 'SmoothL1Loss'])


    args = parser.parse_args()

    # run(args)