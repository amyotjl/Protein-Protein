import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from scipy import spatial as sp
from Architectures.decoder_conv import Decoder
from Architectures.encoder_conv import Encoder

class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        data_shape: tuple,
        in_channels,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        conv_k = 3, conv_s=1, conv_p=1, pool_k=0, pool_s=1,
        use_batch_norm=False, dropout_rate = 0,
        activation_function = nn.ReLU, optimizer = optim.Adagrad, loss_func = nn.MSELoss(), lr= 0.0001,
        random_matrices_orth = None
    ):
        super().__init__()

        self.random_matrices_orth = random_matrices_orth
        self.in_channels = in_channels
        self.lr = lr
        self.data_shape = data_shape
        self.conv_k = conv_k
        self.conv_s = conv_s
        self.conv_p = conv_p
        self.pool_k = pool_k
        self.pool_s = pool_s
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.activation_function = activation_function
        out_dim = data_shape[1]

        for _ in range(len(in_channels)-1):
            out_dim = self.compute_dim(out_dim, self.conv_k, self.conv_s, self.conv_p, self.pool_k, self.pool_s)

        in_channels_reversed = in_channels[::-1]
        self.encoder = encoder_class(out_dim, in_channels, latent_dim, conv_k, conv_s, conv_p, pool_k, pool_s, use_batch_norm, dropout_rate, activation_function)
        self.decoder = decoder_class(out_dim, in_channels_reversed, latent_dim, conv_k, conv_s, conv_p, pool_k, pool_s, use_batch_norm, dropout_rate, activation_function)
        self.out_dim = out_dim

        self.validation_acc = []
        self.validation_acc_epoch = []
        self.training_epoch_mean = []
        self.current_training_epoch_loss = []
        self.training_step_loss = []
        self.validation_epoch_mean = []
        self.current_validation_epoch_loss = []
        self.validation_step_loss = []

        self.kdTree = sp.KDTree(torch.stack([vec for vec in random_matrices_orth.values()]))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        x, labels = batch  # We do not need the labels
        preds = self.forward(x)
        loss = self.loss_func(preds, x)
        return loss, preds, labels

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss,_,_ = self._get_reconstruction_loss(batch)
        self.current_training_epoch_loss.append(loss.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._get_reconstruction_loss(batch)
        self.current_validation_epoch_loss.append(loss.detach().cpu())
        self.validation_acc_epoch.append(self.batch_correct_reconstructed_amino_acid(labels, preds.detach().cpu())[1])

    def on_train_epoch_end(self):
        epoch_loss = np.mean(self.current_training_epoch_loss)
        self.training_step_loss += self.current_training_epoch_loss
        self.training_epoch_mean.append(epoch_loss)
        self.log("train_epoch_loss", epoch_loss, sync_dist=True)
        self.current_training_epoch_loss.clear()

    def on_validation_epoch_end(self):
        epoch_loss = np.mean(self.current_validation_epoch_loss)
        self.validation_step_loss += self.current_validation_epoch_loss
        self.validation_epoch_mean.append(epoch_loss)
        self.log("validation_epoch_loss", epoch_loss, sync_dist=True)
        self.validation_acc.append(np.average(self.validation_acc_epoch))
        self.validation_acc_epoch.clear()
        self.current_validation_epoch_loss.clear()

    def compute_dim(self, dim, conv_k, conv_s, conv_p, pool_k, pool_s):
        dim = np.floor(((dim - conv_k + 2 * conv_p)/conv_s)+1)
        return dim

    def batch_correct_reconstructed_amino_acid(self, sequences, output):
        closest = self.kdTree.query(output)[1]
        aminoacid = list(self.random_matrices_orth.keys())
        correct_aa = 0
        reconstructed_pair = []
        for idx, seq in enumerate(sequences):
            reconstructed = [aminoacid[i] for i in closest[idx]]
            seq = list(seq.ljust(self.data_shape[0], '_'))
            reconstructed_pair.append((seq, reconstructed))
            correct_aa += sum(x == y for x, y in zip(reconstructed, seq))
        accuracy = correct_aa/ (len(sequences)*self.data_shape[0])
        return correct_aa, accuracy, reconstructed_pair
    