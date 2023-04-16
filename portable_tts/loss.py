# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import torch
from torch.nn import functional as F


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def duration_loss(log_duration_prediction, duration_target, mask):
    l = F.mse_loss(log_duration_prediction.squeeze(1),
                   torch.log(duration_target + 1),
                   reduction='sum')
    l = l / mask.sum()
    return l


def f0_loss(pitch_prediction, pitch_target, mask):
    l = F.l1_loss(pitch_prediction,
                  torch.log(pitch_target + 1),
                  reduction='sum')
    l = l / mask.sum()
    return l