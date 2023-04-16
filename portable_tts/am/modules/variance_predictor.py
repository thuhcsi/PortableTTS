# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

from torch import nn


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, input_dim, n_conv_filter, conv_kernel_size, dropout):
        super(VariancePredictor, self).__init__()
        self.input_dim = input_dim
        self.n_conv_filter = n_conv_filter
        self.conv_kernel_size = conv_kernel_size
        self.dropout = dropout

        self.conv_layer = nn.Sequential(
            Conv(
                self.input_dim,
                self.n_conv_filter,
                kernel_size=self.conv_kernel_size,
            ),
            nn.ReLU(),
            nn.LayerNorm(self.n_conv_filter),
            nn.Dropout(self.dropout),
            Conv(
                self.n_conv_filter,
                self.n_conv_filter,
                kernel_size=self.conv_kernel_size,
            ),
            nn.ReLU(),
            nn.LayerNorm(self.n_conv_filter),
            nn.Dropout(self.dropout),
        )

        self.linear_layer = nn.Linear(self.n_conv_filter, 1)

    def forward(self, x, x_mask=None):
        """
        x: (b,t,d)
        x_mask: (b,t,1)
        ---
        output: (b,t)
        """
        out = self.conv_layer(x)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if x_mask is not None:
            x = x * x_mask
        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
    ):
        """

        Args:
            in_channels: dimension of input
            out_channels: dimension of output
            kernel_size: size of kernel
            bias: boolean. if True, bias is included.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding='same',
            bias=bias,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        return x
