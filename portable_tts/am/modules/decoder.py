import torch

from torch import nn

from .encoder import ConvReluNorm, Encoder
from utils import mask

class MelDecoder(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 window_size,
                 spk_emb_dim=None,
                 n_spks=1):
        super().__init__()
        self.n_spks = n_spks

        self.prenet = ConvReluNorm(input_dim +
                                   spk_emb_dim if n_spks > 1 else input_dim,
                                   hidden_channels,
                                   hidden_channels,
                                   kernel_size=5,
                                   n_layers=3,
                                   p_dropout=0.5)

        self.encoder = Encoder(hidden_channels,
                               filter_channels,
                               n_heads,
                               n_layers,
                               kernel_size,
                               window_size=window_size)

    def forward(self, x, x_lengths, spk=None):
        """_summary_

        Args:
            x (_type_): shape (b,d,ty)
            x_lengths (_type_): shape (b)

        """
        x_mask = mask.get_content_mask(x_lengths).unsqueeze(1) # (b,1,t)
        if self.n_spks > 1:
            x = torch.cat([x, spk.unsqueeze(-1).repeat(1, 1, x.shape[-1])],
                          dim=1)
        x = self.prenet(x, x_mask)
        x = self.encoder(x, x_mask)  # (b,d,t)
        return x, x_mask