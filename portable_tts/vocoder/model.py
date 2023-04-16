# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d
from torch.nn.utils import weight_norm, remove_weight_norm
from .modules import generator, discriminator


class Generator(torch.nn.Module):

    def __init__(self, input_dim, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_initial_channel, upsample_rates,
                 upsample_kernel_sizes, gen_istft_n_fft, resblock='resblock1'):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(input_dim, upsample_initial_channel, 7, 1, padding=3))
        resblock = (generator.ResBlock1
                    if resblock == 'resblock1' else generator.ResBlock2)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(upsample_initial_channel // (2**i),
                                    upsample_initial_channel // (2**(i + 1)),
                                    k,
                                    u,
                                    padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(
            Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(generator.init_weights)
        self.conv_post.apply(generator.init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, generator.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])

        return spec, phase

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            discriminator.DiscriminatorP(2),
            discriminator.DiscriminatorP(3),
            discriminator.DiscriminatorP(5),
            discriminator.DiscriminatorP(7),
            discriminator.DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            discriminator.DiscriminatorS(use_spectral_norm=True),
            discriminator.DiscriminatorS(),
            discriminator.DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2),
             AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs