# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import torch


def hz2mel(x):
    return 2595. * torch.log10(1. + x / 700.)


def mel2hz(x):
    return 700 * (10**(x / 2595.) - 1)


if __name__ == '__main__':
    x = torch.randn(10).abs() * 300
    y = hz2mel(x) / 500
    x_hat = mel2hz(y*500)
    print(x)
    print(y)
    print(x_hat)
    print((x - x_hat).abs().sum())
