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
import shutil
from IPython.display import clear_output
from scipy import spatial as sp
from Architectures.decoder_conv import Decoder
from Architectures.encoder_conv import Encoder
from Architectures.autoencoder_conv_pl import Autoencoder
import json

# train, test = get_input_seqs_dataloader('data/AA_orth.joblib',shuffle=False, train_size=0.5, batch_size=64)
# random_matrices_orth = joblib.load('data/AA_random_matrices_orth.joblib')

models = {}
path = "models/K-folds"
for model in os.listdir(path):
    if not model.endswith('txt'):
        models[model] = joblib.load("{}/{}".format(path,model))

names = {}
for item in models:
    name = item[:-2]
    if name not in names:
        names[name]=[]
    names[name].append(models[item])

best = []

for parent, elems in names.items():
    last = []
    for i,item in enumerate(elems):
        last.append(item.validation_acc[-1])
    best.append((parent,np.average(last), last))

best.sort(key=lambda x: x[1])
unused_model = [b[0] for b in best[:-1]]
for model in os.listdir(path):
    if model[:-2] in unused_model:
        os.rename(os.path.join(path, model), os.path.join('models/Unused', model))
with open('{}/all_avg.txt'.format(path), 'w') as f:
    for i in best:
        f.write("{} - {} - {}{}".format(i[1], i[0], i[2],'\n'))
