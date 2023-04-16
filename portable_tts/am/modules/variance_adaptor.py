# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import math

import torch
from torch import nn

from . import variance_predictor, length_regulator
from utils import mask


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, input_dim: int, n_conv_filter: int,
                 conv_kernel_size: int, dropout: float, n_pitch_bin: int,
                 pitch_min, pitch_max) -> None:
        """Initializing variance adaptor.

        Args:
            input_dim (int): Dimension of input features.
            n_conv_filter (int): Number of convolution filters for each variance
            predictor.
            conv_kernel_size (int): Kernel size for all convolution layers in
            all variance predictors.
            dropout (float): Dropout of variance predictors.
            pitch_min (float): Minimum of pitch to construct pitch bins.
            pitch_max (float): Maximum of pitch to construct pitch bins.
            pitch_mean (float): Mean of pitch to construct pitch bins.
            pitch_sigma (float): Standard deviation of pitch to normalize
            minimum and maximum of pitch.
            energy_min (float): Minimum of energy to construct energy bins.
            energy_max (float): Maximum of energy to construct energy bins.
            energy_mean (float): Mean of energy to construct energy bins.
            energy_sigma (float): Standard deviation of energy to normalize
            minimum and maximum of energy.
            n_pitch_bin (int): Number of pitch bins for pitch quantization.
            n_energy_bin (int): Number of energy bins for energy quantization.
        """
        super().__init__()
        self.duration_predictor = variance_predictor.VariancePredictor(
            input_dim, n_conv_filter, conv_kernel_size, dropout)
        self.pitch_predictor = variance_predictor.VariancePredictor(
            input_dim, n_conv_filter, conv_kernel_size, dropout)
        self.length_regulator = length_regulator.LengthRegulator()

        self.pitch_bins = nn.Parameter(
            torch.linspace(math.log(pitch_min + 1), math.log(pitch_max + 1),
                           n_pitch_bin - 1),
            requires_grad=False,
        )

        self.pitch_emb = nn.Embedding(n_pitch_bin, input_dim)

    def get_pitch_embedding(self, x: torch.Tensor, x_mask: torch.Tensor,
                            pitch_target):
        """Getting pitch/energy predictions and embeddings.

        This function predicts pitch/energy from input features and generates
        pitch/energy embeddings. If pitch/energy target is given, it will be
        quantized by pitch/energy bins and converted to corresponding
        embeddings. Otherwise, predicted pitch/energy will be used to produced
        corresponding embeddings.

        Args:
            x (torch.Tensor): Input feature.
            x_mask (torch.Tensor): Mask for x. If x_mask[i,j] is True, x[i,j]
            will be set to zero in pitch/energy predictor.
            predictor (nn.Module): Pitch/energy predictor.
            bins (nn.Parameter): Pitch/energy bins for quantization.
            embedding_table (nn.Module): Pitch/energy embedding table to convert
            quantized value to embeddings.
            target (Optional[torch.Tensor]): Pitch/energy target.
            control (float, optional): Pitch/energy manipulation factor.
            Defaults to 1.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Prediction of pitch/energy,
            pitch/energy embeddings.
        """
        pitch_prediction = self.pitch_predictor(x, x_mask)
        if pitch_target is not None:
            pitch_embedding = self.pitch_emb(
                torch.bucketize(torch.log(pitch_target + 1), self.pitch_bins))
        else:
            pitch_embedding = self.pitch_emb(
                torch.bucketize(pitch_prediction, self.pitch_bins))
        return pitch_prediction, pitch_embedding

    def forward(self,
                x: torch.Tensor,
                x_type,
                duration_target,
                pitch_target,
                d_control: float = 1.0):
        """Predicting pitch, energy and duration and extending input phoneme
        sequences according to duration.

        Args:
            x (torch.Tensor): Input phoneme sequences.
            x_mask (torch.Tensor): Mask for x. If x_mask[i,j] is True, x[i,j]
            will be set to zero in variance predictor.
            duration_target (Optional[torch.Tensor], optional): Ground truth
            duration. Defaults to None.
            pitch_target (Optional[torch.Tensor], optional): Ground truth pitch.
            If this is not provided, the model will use predicted pitch.
            Defaults to None.
            energy_target (Optional[torch.Tensor], optional): Ground truth
            energy. If this is not provided, the model will use predicted
            energy. Defaults to None.
            p_control (float, optional): Pitch manipulation factor. Defaults to
            1.0.
            e_control (float, optional): Energy manipulation factor. Defaults to
            1.0.
            d_control (float, optional): Duration manipulation factor. Defaults
            to 1.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor]: Output of variance adapter, mel-spectrogram length,
            pitch prediction, energy prediction and duration prediction in log
            domain.
        """
        x, x_len = self.length_regulator(x.permute(0, 2, 1), x_type)  # (b,t,d)
        x_mask = mask.get_content_mask(x_len).unsqueeze(2)  # (b,t,1)

        log_duration_prediction = self.duration_predictor(x, x_mask)

        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, x_mask, pitch_target)
        if duration_target is not None:
            output, output_len = self.length_regulator(x + pitch_embedding,
                                                       duration_target)
        else:
            output, output_len = self.length_regulator(
                x + pitch_embedding,
                torch.round(torch.exp(log_duration_prediction) -
                            1).clamp(min=0) * d_control)

        return (output, output_len, pitch_prediction, log_duration_prediction,
                x_mask)
