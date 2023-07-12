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

train_loader, test_loader = get_input_seqs_dataloader('data/AA_orth_500.joblib', shuffle=True, train_size=1, batch_size=1)

model = joblib.load('models/K-folds/1024_[500, 512, 256, 128]_64_Tanh_AdamW_MSELoss_0_False_3_1_2')

outputs = []
seqs = []
for batch in train_loader:
    data, seq = batch

    output = model.encoder(data)
    outputs.append(output)
    seqs.append(seq)

joblib.dump({"inputs":torch.stack(outputs), "sequences":seqs}, 'latent_dim_500.joblib')

