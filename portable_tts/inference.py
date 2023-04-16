# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import pathlib
import argparse

import torch
from torch.utils.data import DataLoader
from scipy.io import wavfile
import tqdm

from portable_tts import dataset, mel, model
from utils import plot, config


def inference_portable_tts(checkpoint_path, output_dir, datalist_path, cfg,
                           cuda, max_wav_value):
    train_dataset = dataset.PortableTTSDataset(
        datalist_path, cfg.dataset.phn2id_path, cfg.dataset.spk2id_path,
        cfg.dataset.special_tokens_path, cfg.dataset.sr,
        cfg.dataset.preemphasis, cfg.dataset.n_fft, cfg.dataset.win_size,
        cfg.dataset.hop_size, cfg.dataset.n_mel, cfg.dataset.mel_f_min,
        cfg.dataset.min_level_db, cfg.dataset.ref_level_db,
        cfg.dataset.pitch_min, cfg.dataset.pitch_max)
    collate_fn = dataset.PortableTTSCollateFn()
    train_loader = DataLoader(train_dataset,
                              num_workers=cfg.PortableTTS.train.num_workers,
                              shuffle=False,
                              collate_fn=collate_fn,
                              batch_size=1)
    portable_tts_model = model.PortableTTS.build(cfg.PortableTTS.model)
    mel_layer = mel.MelspectrogramLayer(cfg.dataset.sr, cfg.dataset.n_fft,
                                        cfg.dataset.hop_size,
                                        cfg.dataset.win_size,
                                        cfg.dataset.n_mel,
                                        cfg.dataset.mel_f_min)
    if cuda:
        portable_tts_model = portable_tts_model.cuda()

    portable_tts_model.get_num_params()

    (epoch, step, model_state_dict, _, _, _, _, _,
        _) = torch.load(checkpoint_path, map_location='cpu')
    portable_tts_model.load_state_dict(model_state_dict)
    print(f'loading checkpoint from epoch {epoch}, step {step}')

    output_dir = pathlib.Path(output_dir) / f'e{epoch}_s{step}'
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        portable_tts_model.eval()
        for batch in tqdm.tqdm(train_loader, total=len(train_dataset)):
            phn_idx = batch['phn_idx']
            phn_length = batch['phn_length']
            duration_target = batch['duration']
            wav_target = batch['audio']
            sample_id = batch['sample_id']
            x_type = batch['type']
            pitch_target = batch['f0']
            if cuda:
                phn_idx = phn_idx.cuda()
                phn_length = phn_length.cuda()
                x_type = x_type.cuda()
            (_, _, _, log_pitch_prediction, log_duration_prediction, _, _,
             wav_prediction,
             _) = portable_tts_model.infer(phn_idx, phn_length, x_type)
            if cuda:
                log_pitch_prediction = log_pitch_prediction.cpu()
                log_duration_prediction = log_duration_prediction.cpu()
                wav_prediction = wav_prediction.cpu()
            duration_prediction = torch.ceil(
                torch.exp(log_duration_prediction) - 1).squeeze(1)
            pitch_prediction = (torch.exp(log_pitch_prediction) - 1).squeeze(1)
            wav_prediction_mel = mel_layer(wav_prediction.squeeze(1))
            wav_target_mel = mel_layer(wav_target.squeeze(1))
            for (p_pred, p_tgt, d_pred, d_tgt, mel_pred, mel_tgt, w_pred,
                 w_tgt, phn_seq, phn_type,
                 id) in zip(pitch_prediction, pitch_target,
                            duration_prediction, duration_target,
                            wav_prediction_mel, wav_target_mel, wav_prediction,
                            wav_target, batch['phn'], batch['type'],
                            sample_id):
                phn_dur_infer = get_phn_dur_list(phn_seq, phn_type, d_pred)
                phn_dur_tgt = get_phn_dur_list(phn_seq, phn_type, d_tgt)
                plot.plot_mel_dur([mel_tgt.T, mel_pred.T],
                                  [f'ground truth {id}', f'inference'],
                                  [phn_dur_tgt, phn_dur_infer],
                                  path=output_dir / f'{id}_mel.png')
                plot.plot_f0([p_tgt, p_pred],
                             [f'ground truth {id}', f'inference'],
                             path=output_dir / f'{id}_f0.png')
                w_pred = (w_pred / (w_pred.abs().max()) *
                          max_wav_value)[0].numpy().astype('int16')
                w_tgt = (w_tgt / (w_tgt.abs().max()) *
                         max_wav_value).numpy().astype('int16')
                wavfile.write(output_dir / f'{id}_gt.wav', cfg.dataset.sr,
                              w_tgt)
                wavfile.write(output_dir / f'{id}_pred.wav', cfg.dataset.sr,
                              w_pred)


def get_phn_dur_list(phn_seq, token_type, duration):
    """
    phn_seq: list
    token_type: list
    duration: list
    """
    dur_list = []
    k = 0
    for phn, type in zip(phn_seq, token_type):
        if type == 1:
            dur_list.append((phn, duration[k]))
            k += 1
    return dur_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--ckpt_path', type=str, help='path to checkpoint')
    parser.add_argument('--datalist_path', type=str, help='path to datalist')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='use cuda for inference')
    parser.add_argument('--output_path',
                        type=str,
                        help='path to output inference samples')
    parser.add_argument('--max_wav_value', type=int, default=32767)
    return parser.parse_args()


def main():
    args = get_args()
    cfg = config.from_file(args.config)
    inference_portable_tts(args.ckpt_path, args.output_path,
                           args.datalist_path, cfg, args.cuda,
                           args.max_wav_value)


if __name__ == "__main__":
    main()