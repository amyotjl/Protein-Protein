import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, out_dim,in_channels, latent_dim: int, conv_k = 3, conv_s=1, conv_p=1, pool_k=0, pool_s=1,
                 use_batch_norm=False, dropout_rate = 0,
                 activation_function = nn.ReLU):
        super().__init__()
        self.conv_k = conv_k
        self.conv_s = conv_s
        self.conv_p = conv_p
        self.pool_k = pool_k
        self.pool_s = pool_s
        self.in_channels = in_channels
        self.use_batch_norm = use_batch_norm
        self.layers = nn.ModuleList()


        for i in range(len(self.in_channels)-1):
            self.layers.append(nn.Conv1d(self.in_channels[i], self.in_channels[i+1],  kernel_size=self.conv_k, stride=self.conv_s, padding=self.conv_p))
            self.layers.append(activation_function())
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(self.in_channels[i+1]))
            self.layers.append(nn.Dropout(p=dropout_rate))


        self.conv_net = nn.Sequential(*self.layers)
        self.fc = nn.Linear(int(out_dim* self.in_channels[-1]), latent_dim)


    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
