import torch
from torch import nn

from .am.model import LightTransformer
from .vocoder.model import Generator
from . import common


class PortableTTS(nn.Module):

    def __init__(self, n_vocab, n_spks, spk_emb_dim, enc_hidden_dim,
                 enc_filter_channels, enc_self_attn_n_heads,
                 enc_self_attn_n_layers, enc_self_attn_kernel_size,
                 enc_window_size, dec_hidden_dim, dec_filter_channels,
                 dec_self_attn_n_heads, dec_self_attn_n_layers,
                 dec_self_attn_kernel_size, dec_window_size, va_n_conv_filter,
                 va_conv_kernel_size, va_dropout, va_n_pitch_bin,
                 gen_resblock_kernel_sizes, gen_resblock_dilation_sizes,
                 gen_upsample_initial_channel, gen_upsample_rates,
                 gen_upsample_kernel_size, gen_istft_n_fft, gen_istft_hop_size,
                 segment_size, pitch_min, pitch_max) -> None:
        super().__init__()

        self.segment_size = segment_size
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        self.am = LightTransformer(
            n_vocab, n_spks, spk_emb_dim, enc_hidden_dim, enc_filter_channels,
            enc_self_attn_n_heads, enc_self_attn_n_layers,
            enc_self_attn_kernel_size, enc_window_size, dec_hidden_dim,
            dec_filter_channels, dec_self_attn_n_heads, dec_self_attn_n_layers,
            dec_self_attn_kernel_size, dec_window_size, va_n_conv_filter,
            va_conv_kernel_size, va_dropout, va_n_pitch_bin, pitch_min,
            pitch_max)
        self.vocoder = Generator(dec_hidden_dim, gen_resblock_kernel_sizes,
                                 gen_resblock_dilation_sizes,
                                 gen_upsample_initial_channel,
                                 gen_upsample_rates, gen_upsample_kernel_size,
                                 gen_istft_n_fft)
        self.istft_win = nn.Parameter(torch.hann_window(gen_istft_n_fft),
                                      requires_grad=False)

    def forward(self, x, x_length, x_type, duration, pitch, spk=None):
        (dec_output, dec_output_len, dec_output_mask, pitch_prediction,
         log_duration_prediction, va_mask) = self.am(x, x_length, x_type,
                                                     duration, pitch, spk)
        dec_output_slice, dec_output_idx = common.rand_slice_segments(
            dec_output, dec_output_len, self.segment_size)

        spec_slice, phase_slice = self.vocoder(dec_output_slice)
        wav_slice = self.istft(spec_slice, phase_slice)
        return (dec_output, dec_output_len, dec_output_mask, pitch_prediction,
                log_duration_prediction, dec_output_slice, dec_output_idx,
                spec_slice, phase_slice, wav_slice, va_mask)

    def remove_weight_norm(self):
        self.vocoder.remove_weight_norm()

    def infer(self, x, x_length, x_type, spk=None):
        (dec_output, dec_output_len, dec_output_mask, pitch_prediction,
         log_duration_prediction, va_mask) = self.am(x,
                                                     x_length,
                                                     x_type,
                                                     spk=spk)
        spec, phase = self.vocoder(dec_output)
        wav = self.istft(spec, phase)
        return (dec_output, dec_output_len, dec_output_mask, pitch_prediction,
                log_duration_prediction, spec, phase, wav, va_mask)

    def istft(self, spec, phase):
        inverse_transform = torch.istft(
            spec * torch.exp(phase * 1j),
            self.gen_istft_n_fft,
            self.gen_istft_hop_size,
            self.gen_istft_n_fft,
            self.istft_win,
        )
        return inverse_transform.unsqueeze(1)

    def get_num_params(self):

        def get_num_parameters(m):
            return sum([p.numel() for p in m.parameters()])

        print(f'TextEncoder:{get_num_parameters(self.am.encoder)}')
        print(f'Decoder:{get_num_parameters(self.am.decoder)}')
        print(
            f'VarianceAdopter:{get_num_parameters(self.am.variance_adaptor)}')
        print(f'Vocoder:{get_num_parameters(self.vocoder)}')

    @classmethod
    def build(cls, config):
        return cls(config.n_vocab, config.n_spks, config.spk_emb_dim,
                   config.enc_hidden_dim, config.enc_filter_channels,
                   config.enc_self_attn_n_heads, config.enc_self_attn_n_layers,
                   config.enc_self_attn_kernel_size, config.enc_window_size,
                   config.dec_hidden_dim, config.dec_filter_channels,
                   config.dec_self_attn_n_heads, config.dec_self_attn_n_layers,
                   config.dec_self_attn_kernel_size, config.dec_window_size,
                   config.va_n_conv_filter, config.va_conv_kernel_size,
                   config.va_dropout, config.va_n_pitch_bin,
                   config.gen_resblock_kernel_sizes,
                   config.gen_resblock_dilation_sizes,
                   config.gen_upsampe_initial_channel,
                   config.gen_upsample_rates, config.gen_upsample_kernel_size,
                   config.gen_istft_n_fft, config.gen_istft_hop_size,
                   config.segment_size, config.pitch_min, config.pitch_max)
