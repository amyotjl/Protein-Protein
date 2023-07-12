import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, out_dim, in_channels, latent_dim: int, conv_k = 3, conv_s=1, conv_p=1, pool_k=2, pool_s=2,
                 use_batch_norm=False, dropout_rate = 0,
                 activation_function = nn.ReLU, final_dim = 22):
        super().__init__()
        self.conv_k = conv_k
        self.conv_s = conv_s
        self.conv_p = conv_p
        self.pool_k = pool_k
        self.pool_s = pool_s

        self.in_channels = in_channels
        self.layers = nn.ModuleList()
        self.latent_dim = latent_dim
        current_final_dim = out_dim

        for i in range(len(self.in_channels)-1):
            
            current_final_dim = self.conv_transpose_dim(current_final_dim, conv_k, conv_s, conv_p)
            self.layers.append(nn.ConvTranspose1d (self.in_channels[i], self.in_channels[i+1],  kernel_size=self.conv_k, stride=self.conv_s, padding=self.conv_p, output_padding= 0 if i != len(self.in_channels)-2 else int(final_dim - current_final_dim)))
            self.layers.append(activation_function())
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(self.in_channels[i+1]))
            self.layers.append(nn.Dropout(p=dropout_rate))
            # print(current_final_dim)

        self.layers.append(nn.Tanh())
        self.conv_net = nn.Sequential(*self.layers)
        self.fc = nn.Linear(latent_dim, int(out_dim*in_channels[0]))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.in_channels[0], -1)
        x = self.conv_net(x)
        return x
    
    def conv_transpose_dim (self, dim, k, s, p, d=1):
        return ((dim - 1) * s) - (2 * p) + d * (k - 1) + 1
    