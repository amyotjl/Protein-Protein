# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from matplotlib import pyplot as plt
import os
import gc
from load_dataset import get_input_seqs_dataloader
import utils
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import shutil
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from scipy import spatial as sp
from Architectures.decoder_conv import Decoder
from Architectures.encoder_conv import Encoder
from Architectures.autoencoder_conv_pl import Autoencoder
import sys


# # %%

# data = joblib.load('data/AA_orth.joblib')

# data['sequences'] = np.array(data['sequences'])

random_matrices = joblib.load('data/AA_random_matrices_orth.joblib')
train_loader, test_loader = get_input_seqs_dataloader('data/AA_orth_500.joblib', shuffle=True, train_size=0.8, batch_size=64, num_workers=8)
# %%
config = {
    'output_shape': [(500,22)],
    'epochs': [30],
    'bs':[64],
    'latent_dim':[1024],
    'channels':[[500, 512, 256, 128]],
    # 'channels':[[1965,512,256],[1965,512,64], [1965, 512, 256, 128], [1965,512,256,128,64]],
    'conv_k': [3],
    'conv_s': [1],
    'conv_p': [2],
    # 'pool_k': range(2,4),
    # 'pool_s': range(0,2),
    'activation_func': [nn.Tanh],
    'loss_func': [nn.MSELoss()],
    'optimizer' : [optim.AdamW],
    'dropout_rate' : [0],
    'batch_normalization' : [False]
}

# %%
kf = KFold(5, shuffle=True)


torch.set_float32_matmul_precision('medium')
for conf in utils.get_combinations(config):
        gc.collect()
        file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                conf['latent_dim'], conf["channels"], conf['bs'], conf["activation_func"].__name__,
                                conf["optimizer"].__name__, conf["loss_func"].__class__.__name__, conf["dropout_rate"], conf["batch_normalization"],
                                conf['conv_k'],conf['conv_s'],conf['conv_p']
                                )
        # if (not any(file_name in f for f in os.listdir('models/K-folds/'))) and (not any(file_name in f for f in os.listdir('models/Unused/'))):
        print(conf)
        # for i, (train_index, test_index) in enumerate(kf.split(data['inputs'], data['sequences'])):
        # train_loader = DataLoader(list(zip(data['inputs'][train_index], data['sequences'][train_index])), batch_size =  conf['bs'], shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
        # test_loader = DataLoader(list(zip(data['inputs'][test_index], data['sequences'][test_index])), batch_size =  conf['bs'], shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
        gc.collect()
        early_stopping = EarlyStopping(monitor="validation_epoch_loss", patience=3, min_delta=0.0001, verbose=True)
        model = Autoencoder(conf['output_shape'], conf['channels'], conf['latent_dim'], conv_k=conf['conv_k'],conv_s=conf['conv_s'],conv_p=conf['conv_p'], optimizer=conf['optimizer'], dropout_rate=conf['dropout_rate'], activation_function=conf['activation_func'], loss_func=conf['loss_func'], random_matrices_orth=random_matrices)
        trainer = pl.Trainer(devices=1, max_epochs=conf['epochs'], callbacks=[early_stopping], num_sanity_val_steps=0, check_val_every_n_epoch=1, enable_progress_bar=False)
        tuner = pl.tuner.Tuner(trainer)
        lr_finder = tuner.lr_find(model,train_dataloaders = train_loader, val_dataloaders=test_loader, update_attr=True)
        trainer.fit(model, train_loader, test_loader)
        joblib.dump(model, 'models/K-folds/{}'.format(file_name))
        # shutil.rmtree("lightning_logs")
        # os.makedirs('lightning_logs')
