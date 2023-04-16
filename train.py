# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import argparse
import os

import torch
import torch.multiprocessing as mp

from portable_tts.train import train_portable_tts
from utils import config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="Path to config.yaml", required=True
    )
    parser.add_argument(
        "-p", "--port", type=str, help="Port for distributed training", default="10086"
    )
    return parser.parse_args()


def main():
    args = get_args()
    cfg = config.from_file(args.config)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

    n_gpus = torch.cuda.device_count()
    print(f"training model PortableTTS, config file {args.config}")
    mp.spawn(train_portable_tts, args=(n_gpus, cfg), nprocs=n_gpus)


if __name__ == "__main__":
    main()