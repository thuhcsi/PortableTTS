# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import torch
from torch import nn

from .modules import encoder, decoder, variance_adaptor


class LightTransformer(nn.Module):

    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        enc_hidden_dim,
        enc_filter_channels,
        enc_self_attn_n_heads,
        enc_self_attn_n_layers,
        enc_self_attn_kernel_size,
        enc_window_size,
        dec_hidden_dim,
        dec_filter_channels,
        dec_self_attn_n_heads,
        dec_self_attn_n_layers,
        dec_self_attn_kernel_size,
        dec_window_size,
        va_n_conv_filter,
        va_conv_kernel_size,
        va_dropout,
        va_n_pitch_bin,
        pitch_min,
        pitch_max,
    ):
        super().__init__()

        if n_spks > 1:
            self.spk_emb = nn.Embedding(n_spks, spk_emb_dim)
        self.n_spks = n_spks
        self.encoder = encoder.TextEncoder(n_vocab,
                                           enc_hidden_dim,
                                           enc_filter_channels,
                                           enc_self_attn_n_heads,
                                           enc_self_attn_n_layers,
                                           enc_self_attn_kernel_size,
                                           enc_window_size,
                                           spk_emb_dim=spk_emb_dim,
                                           n_spks=n_spks)
        self.decoder = decoder.MelDecoder(enc_hidden_dim,
                                          dec_hidden_dim,
                                          dec_filter_channels,
                                          dec_self_attn_n_heads,
                                          dec_self_attn_n_layers,
                                          dec_self_attn_kernel_size,
                                          dec_window_size,
                                          spk_emb_dim=spk_emb_dim,
                                          n_spks=n_spks)
        self.variance_adaptor = variance_adaptor.VarianceAdaptor(
            enc_hidden_dim, va_n_conv_filter, va_conv_kernel_size, va_dropout,
            va_n_pitch_bin, pitch_min, pitch_max)

    def forward(self,
                x,
                x_lengths,
                x_type,
                duration=None,
                pitch=None,
                spk=None):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            durations : (b,t_x)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        enc_output, enc_output_mask = self.encoder(x, x_lengths,
                                                   spk)  # (b,d,t)
        (dec_input, dec_input_len, pitch_prediction, log_duration_prediction,
         va_mask) = self.variance_adaptor(enc_output, x_type, duration, pitch)
        dec_input = dec_input.permute(0,2,1)
        dec_output, dec_output_mask = self.decoder(dec_input, dec_input_len,
                                                   spk)

        return (dec_output, dec_input_len, dec_output_mask, pitch_prediction,
                log_duration_prediction, va_mask)
