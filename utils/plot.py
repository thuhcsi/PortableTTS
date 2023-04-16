# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import math

import numpy as np
from matplotlib import pyplot as plt


def get_figsize(data, base=3):
    max_height = max([mel.shape[1] for mel in data])
    max_width = max([mel.shape[0] for mel in data])
    height = len(data) * base
    width = base * max(math.floor(max_width / max_height), 1)
    return width, height


def plot_mel(data, titles=None, dpi=200, path=None):
    fig, axes = plt.subplots(len(data),
                             1,
                             squeeze=False,
                             dpi=dpi,
                             figsize=get_figsize(data))
    if titles is None:
        titles = [None] * len(data)
    max_x = max([mel.shape[0] for mel in data])

    for i, mel in enumerate(data):
        # mel: (t,d)
        im = axes[i][0].imshow(mel.T,
                               origin="lower",
                               aspect='auto',
                               interpolation='none')
        axes[i][0].set_ylim(0, mel.shape[1])
        axes[i][0].set_xlim(0, max_x)
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small")
        plt.colorbar(im, ax=axes[i][0])
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
        plt.close()
    else:
        return fig


def plot_mel_dur(data, titles=None, durations=None, dpi=200, path=None):
    fig, axes = plt.subplots(len(data),
                             1,
                             squeeze=False,
                             dpi=dpi,
                             figsize=get_figsize(data))
    if titles is None:
        titles = [None] * len(data)

    for i, mel in enumerate(data):
        # mel: (t,d)
        im = axes[i][0].imshow(mel.T,
                               origin="lower",
                               aspect='auto',
                               interpolation='none')
        axes[i][0].set_ylim(0, mel.shape[1])
        axes[i][0].set_xlim(0, mel.shape[0])
        axes[i][0].set_title(f'{titles[i]} {mel.shape}', fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small")
        axes[i][0].set_xticks(np.array([d[1] for d in durations[i]]).cumsum())
        axes[i][0].set_xticklabels([f'{d[0]}:{d[1]}' for d in durations[i]],
                                   rotation=300)
        plt.colorbar(im, ax=axes[i][0])
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
        plt.close()
    else:
        return fig


def plot_f0(data, titles=None, dpi=200, path=None):
    #print(data)
    def get_f0_size(data):
        b = 3
        max_width = max([len(mel) for mel in data])
        height = len(data) * b
        width = max(math.floor(max_width / 80), 1) * b
        return width, height

    fig, axes = plt.subplots(len(data),
                             1,
                             squeeze=False,
                             dpi=dpi,
                             figsize=get_f0_size(data))
    if titles is None:
        titles = [None] * len(data)

    for i, f0 in enumerate(data):
        axes[i][0].plot(np.arange(1, len(f0) + 1), f0)
        axes[i][0].set_ylim(0, 800)
        axes[i][0].set_xlim(0, len(f0))
        axes[i][0].set_title(f'{titles[i]}', fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small")

    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
        plt.close()
    else:
        return fig
