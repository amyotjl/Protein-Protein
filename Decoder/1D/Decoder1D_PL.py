import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import joblib
import os
from load_dataset import get_dataloader
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from utils import get_combinations
import gc

class ConvNet1D(pl.LightningModule):
    def __init__(self, input_size, output_size, in_channels,
                 conv_k = 3, conv_s=1, conv_p=1, pool_k=2, pool_s=2,
                 use_batch_norm=False, dropout_rate = 0,
                 activation_function = nn.ReLU(), optimizer = optim.Adagrad, loss_func = nn.MSELoss()):
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
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.training_epoch_mean = []
        self.current_training_epoch_loss = []
        self.training_step_loss = []
        self.validation_epoch_mean = []
        self.current_validation_epoch_loss = []
        self.validation_step_loss = []
        self.lr = 0.001

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
    
    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.current_validation_epoch_loss.append(loss.detach().cpu().numpy())
        return loss
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.current_training_epoch_loss.append(loss.detach().cpu().numpy())
        return loss
    
    def _common_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self.forward(data)
        loss = self.loss_func(outputs, labels)
        return loss, outputs, labels
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def on_train_epoch_end(self):
        epoch_loss = np.mean(self.current_training_epoch_loss)
        self.training_step_loss += self.current_training_epoch_loss
        self.training_epoch_mean.append(epoch_loss)
        self.log("train_epoch_loss", epoch_loss, prog_bar=True)
        self.current_training_epoch_loss.clear()

    def on_validation_epoch_end(self):
        epoch_loss = np.mean(self.current_validation_epoch_loss)
        self.validation_step_loss += self.current_validation_epoch_loss
        self.validation_epoch_mean.append(epoch_loss)
        self.log("validation_epoch_loss", epoch_loss, prog_bar=True)
        self.current_validation_epoch_loss.clear()
    
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


def run(args, train_loader, test_loader):
    model = ConvNet1D(args["embeddings_length"],
                      args["output_shape"],
                      args["channels"],
                      args["conv_k"], args["conv_s"], args["conv_p"],
                      args["pool_k"],args["pool_s"],
                      args["batch_normalization"],
                      args["dropout_rate"],
                      args["activation_func"],
                      optimizer=args["optimizer"],
                      loss_func=args["loss_func"]
                      )
    
    early_stopping = EarlyStopping(monitor="validation_epoch_loss", patience=3, min_delta=0.0001, verbose=True, check_on_train_epoch_end=False, mode = "min" )
    ddp = DDPStrategy(process_group_backend="gloo")


    checkpoints_path = "checkpoints/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                        args["channels"], args["activation_func"].__class__.__name__,
                        args["optimizer"].__name__, args["loss_func"].__class__.__name__, args["dropout_rate"], args["batch_normalization"],
                        args["conv_k"], args["conv_s"], args["conv_p"],
                        args["pool_k"],args["pool_s"])

    os.makedirs(checkpoints_path, exist_ok=True)

    checkpoints = ModelCheckpoint(
        dirpath="{}.ckpt".format(checkpoints_path),
        filename="{epoch}",
        every_n_epochs=10,
        save_top_k=-1
    )

    trainer = pl.Trainer(accelerator='gpu', strategy=ddp, devices=1, max_epochs=args["epochs"], callbacks=[early_stopping, checkpoints], num_sanity_val_steps=0)
    
    tuner = pl.tuner.Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_loader, test_loader)
    new_lr = lr_finder.suggestion()
    model.lr = new_lr

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    joblib.dump(model, "{}.joblib".format(checkpoints_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script creates a Conv1D model based on pre-calculated embeddings.')
    parser.add_argument( '-embs','--embeddings_path', type=str, help="Path to the embedding file", default = 'data/Prots_embeddings_1d.joblib')
    parser.add_argument('-rm','--random_matrices', type=str, help="Path to the random matrices file", default = 'data/AA_random_matrices.joblib')
    parser.add_argument('-bs','--batch_size', type=int, default = 256)
    parser.add_argument('-ts','--training_size', type=float, choices = [i/100 for i in range(101)], help = 'Size of the training set. Test size will be the complement. This value must be between 0 and 1.', default = 0.8)
    
    torch.set_float32_matmul_precision('medium')    

    args = parser.parse_args()

    AA_random_matrices = joblib.load(args.random_matrices)
    train_loader, test_loader = get_dataloader(args.embeddings_path, AA_random_matrices, args.batch_size, train_size=args.training_size, shuffle=True)

    # Grid search
    config = {
        'output_shape': [(1965,22)],
        'embeddings_length': [1024],
        'epochs': [25],
        'channels':[[1,2,4,8], [1,4,8]], # Should be changed 
        'conv_k': range(2,5),
        'conv_s': range(1,3),
        'conv_p': range(0,2),
        'pool_k': range(2,4),
        'pool_s': range(1,3),
        'activation_func': [nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU()],
        'loss_func': [nn.MSELoss(), nn.L1Loss()],
        'optimizer' : [optim.Adagrad, optim.AdamW],
        'training_size': [0.85],
        'dropout_rate' : [0, 0.25],
        'batch_normalization' : [True, False]
    }

    for conf in get_combinations(config):
        gc.collect()
        print(conf)
        run(conf, train_loader, test_loader)
