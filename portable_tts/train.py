# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import pathlib
import itertools
import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import dataset, common, loss, mel
from .model import PortableTTS
from .vocoder.model import MultiPeriodDiscriminator, MultiScaleDiscriminator
from utils import plot, config

MODEL_NAME = "portable_tts"


def train_portable_tts(rank, world_size, cfg):
    if rank == 0:
        writer = SummaryWriter(log_dir=cfg.PortableTTS.train.logdir)
        save_dir = pathlib.Path(cfg.PortableTTS.train.logdir) / "ckpt"
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg.to_file(save_dir / "config.yaml")
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    torch.manual_seed(cfg.PortableTTS.train.seed)
    torch.cuda.set_device(rank)
    train_dataset = dataset.PortableTTSDataset(
        cfg.dataset.datalist_path,
        cfg.dataset.phn2id_path,
        cfg.dataset.spk2id_path,
        cfg.dataset.special_tokens_path,
        cfg.dataset.sr,
        cfg.dataset.preemphasis,
        cfg.dataset.n_fft,
        cfg.dataset.win_size,
        cfg.dataset.hop_size,
        cfg.dataset.n_mel,
        cfg.dataset.mel_f_min,
        cfg.dataset.min_level_db,
        cfg.dataset.ref_level_db,
        cfg.dataset.pitch_min,
        cfg.dataset.pitch_max,
    )
    train_sampler = DistributedSampler(train_dataset, world_size, rank)
    collate_fn = dataset.PortableTTSCollateFn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=cfg.PortableTTS.train.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        batch_size=cfg.PortableTTS.train.batch_size,
        sampler=train_sampler,
    )
    portable_tts_model = PortableTTS.build(cfg.PortableTTS.model).to(rank)
    if rank == 0:
        portable_tts_model.get_num_params()
    disc_mpd_model = MultiPeriodDiscriminator().to(rank)
    disc_msd_model = MultiScaleDiscriminator().to(rank)
    mel_layer = mel.MelspectrogramLayer(
        cfg.dataset.sr,
        cfg.dataset.n_fft,
        cfg.dataset.hop_size,
        cfg.dataset.win_size,
        cfg.dataset.n_mel,
        cfg.dataset.mel_f_min,
    ).to(rank)
    ddp_portable_tts_model = DDP(portable_tts_model, device_ids=[rank])
    ddp_disc_mpd = DDP(disc_mpd_model, device_ids=[rank])
    ddp_disc_msd = DDP(disc_msd_model, device_ids=[rank])

    optim_portable_tts = torch.optim.AdamW(
        ddp_portable_tts_model.parameters(),
        cfg.PortableTTS.train.lr,
        cfg.PortableTTS.train.betas,
        cfg.PortableTTS.train.eps,
    )
    optim_disc = torch.optim.AdamW(
        itertools.chain(ddp_disc_mpd.parameters(), ddp_disc_msd.parameters()),
        cfg.PortableTTS.train.lr,
        cfg.PortableTTS.train.betas,
        cfg.PortableTTS.train.eps,
    )
    lr_sch_portable_tts = torch.optim.lr_scheduler.ExponentialLR(
        optim_portable_tts, gamma=cfg.PortableTTS.train.weight_decay
    )
    lr_sch_disc = torch.optim.lr_scheduler.ExponentialLR(
        optim_disc, gamma=cfg.PortableTTS.train.weight_decay
    )
    epoch = 1
    step = 1
    if cfg.PortableTTS.train.ckpt:
        (
            epoch,
            step,
            model_state_dict,
            disc_mpd_state_dict,
            disc_msd_state_dict,
            optim_portable_tts_state_dict,
            optim_disc_state_dict,
            lr_sch_portable_tts_state_dict,
            lr_sch_disc_state_dict,
        ) = torch.load(cfg.PortableTTS.train.ckpt, map_location="cpu")
        ddp_portable_tts_model.module.load_state_dict(model_state_dict)
        ddp_disc_mpd.module.load_state_dict(disc_mpd_state_dict)
        ddp_disc_msd.module.load_state_dict(disc_msd_state_dict)
        optim_portable_tts.load_state_dict(optim_portable_tts_state_dict)
        optim_disc.load_state_dict(optim_disc_state_dict)
        lr_sch_portable_tts.load_state_dict(lr_sch_portable_tts_state_dict)
        lr_sch_disc.load_state_dict(lr_sch_disc_state_dict)
        if rank == 0:
            print(f"resume training from epoch {epoch}, step {step}")
        epoch += 1
        step += 1
    for i in range(epoch, cfg.PortableTTS.train.epoch + 1):
        train_sampler.set_epoch(i)
        ddp_portable_tts_model.train()
        ddp_disc_msd.train()
        ddp_disc_mpd.train()
        for batch in train_loader:
            phn_idx = batch["phn_idx"].to(rank)
            phn_length = batch["phn_length"].to(rank)
            duration_target = batch["duration"].to(rank)
            wav_target = batch["audio"].to(rank)
            sample_id = batch["sample_id"]
            x_type = batch["type"].to(rank)
            pitch_target = batch["f0"].to(rank)

            (
                dec_output,
                dec_output_len,
                dec_output_mask,
                pitch_prediction,
                log_duration_prediction,
                dec_output_slice,
                dec_output_idx,
                spec_slice,
                phase_slice,
                wav_prediction_slice,
                va_mask,
            ) = ddp_portable_tts_model(
                phn_idx,
                phn_length,
                x_type,
                duration_target,
                pitch_target,
            )

            wav_target_slice = common.slice_segments(
                wav_target.unsqueeze(1),
                dec_output_idx * cfg.dataset.hop_size,
                cfg.PortableTTS.model.segment_size * cfg.dataset.hop_size,
            ).to(rank)

            wav_prediction_slice_mel = mel_layer(wav_prediction_slice.squeeze(1))
            wav_target_slice_mel = mel_layer(wav_target_slice.squeeze(1))

            # Discriminator

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = ddp_disc_mpd(
                wav_target_slice, wav_prediction_slice.detach()
            )
            (loss_disc_f, losses_disc_f_r, losses_disc_f_g) = loss.discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = ddp_disc_msd(
                wav_target_slice, wav_prediction_slice.detach()
            )
            (loss_disc_s, losses_disc_s_r, losses_disc_s_g) = loss.discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

            loss_disc_all = loss_disc_s + loss_disc_f
            optim_disc.zero_grad()
            loss_disc_all.backward()
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                itertools.chain(ddp_disc_mpd.parameters(), ddp_disc_msd.parameters()),
                cfg.PortableTTS.train.discriminator_clip_grad_norm,
            )
            optim_disc.step()

            # Generator

            loss_mel = (
                F.l1_loss(wav_target_slice_mel, wav_prediction_slice_mel)
                * cfg.PortableTTS.train.coeff_mel_loss
            )
            loss_dur = loss.duration_loss(
                log_duration_prediction, duration_target, va_mask
            )
            loss_f0 = loss.f0_loss(pitch_prediction, pitch_target, va_mask)
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = ddp_disc_mpd(
                wav_target_slice, wav_prediction_slice
            )
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = ddp_disc_msd(
                wav_target_slice, wav_prediction_slice
            )
            loss_fm_f = (
                loss.feature_loss(fmap_f_r, fmap_f_g)
                * cfg.PortableTTS.train.coeff_feat_match
            )
            loss_fm_s = (
                loss.feature_loss(fmap_s_r, fmap_s_g)
                * cfg.PortableTTS.train.coeff_feat_match
            )
            loss_gen_f, losses_gen_f = loss.generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = loss.generator_loss(y_ds_hat_g)
            loss_gen_s = loss_gen_s * 2
            loss_gen_all = (
                loss_gen_s
                + loss_gen_f
                + loss_fm_s
                + loss_fm_f
                + loss_mel
                + loss_dur
                + loss_f0
            )

            optim_portable_tts.zero_grad()

            loss_gen_all.backward()
            optim_portable_tts.step()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                ddp_portable_tts_model.parameters(),
                cfg.PortableTTS.train.generator_clip_grad_norm,
            )
            optim_portable_tts.step()

            if rank == 0:
                print(
                    f"epoch: {i}, step: {step}, "
                    f"dur loss: {loss_dur.item():.5f}, "
                    f"pitch loss: {loss_f0.item():.5f}, "
                    f"mpd loss: {loss_disc_f.item():.5f}, "
                    f"msd loss: {loss_disc_s.item():.5f}, "
                    f"gen mel loss: {loss_mel.item():.5f}, "
                    f"gen fml mpd: {loss_fm_f.item():.5f}, "
                    f"gen fml msd: {loss_fm_s.item():.5f}, "
                    f"gen adv loss mpd: {loss_gen_f.item():.5f}, "
                    f"gen adv loss msd: {loss_gen_s.item():.5f}, "
                    f"disc grad norm: {grad_norm_d.item():.5f}, "
                    f"gen grad norm: {grad_norm_g.item():.5f}"
                )
                if step and step % cfg.PortableTTS.train.log_interval == 0:
                    writer.add_scalar("dur loss", loss_dur.item(), step)
                    writer.add_scalar("pitch loss", loss_f0.item(), step)
                    writer.add_scalar("mpd loss", loss_disc_f.item(), step)
                    writer.add_scalar("msd loss", loss_disc_s.item(), step)
                    writer.add_scalar("gen mel loss", loss_mel.item(), step)
                    writer.add_scalar("gen fml mpd", loss_fm_f.item(), step)
                    writer.add_scalar("gen fml msd", loss_fm_s.item(), step)
                    writer.add_scalar("gen adv loss mpd", loss_gen_f.item(), step)
                    writer.add_scalar("gen adv loss msd", loss_gen_s.item(), step)
                    writer.add_scalar("disc grad norm", grad_norm_d.item(), step)
                    writer.add_scalar("gen grad norm", grad_norm_g.item(), step)
                if step and step % cfg.PortableTTS.train.log_mel_interval == 0:
                    with torch.no_grad():
                        ddp_portable_tts_model.eval()
                        fig = plot.plot_mel(
                            [
                                wav_target_slice_mel[0].cpu().T.numpy(),
                                wav_prediction_slice_mel[0].cpu().T.numpy(),
                            ],
                            [f"ground truth {sample_id[0]}", "reconstructed"],
                        )
                        writer.add_figure("mel slice", fig, step)

                        phn_dur_train = get_phn_dur_list(
                            batch["phn"][0], x_type[0].cpu(), duration_target[0].cpu()
                        )

                        infer_phn_idx = phn_idx[:1]
                        infer_phn_length = phn_length[:1]
                        infer_wav_target = wav_target[:1]
                        infer_x_type = x_type[:1]
                        infer_pitch = pitch_target[:1]

                        (
                            _,
                            _,
                            _,
                            log_pitch_prediction,
                            log_duration_prediction,
                            _,
                            _,
                            wav_prediction,
                            va_mask,
                        ) = ddp_portable_tts_model.module.infer(
                            infer_phn_idx, infer_phn_length, infer_x_type
                        )
                        duration_prediction = torch.ceil(
                            torch.exp(log_duration_prediction) - 1
                        )
                        pitch_prediction = torch.exp(log_pitch_prediction) - 1
                        wav_prediction_mel = (
                            mel_layer(wav_prediction[0])[0].cpu().T.numpy()
                        )
                        wav_target_mel = (
                            mel_layer(infer_wav_target[0])[0].cpu().T.numpy()
                        )

                        phn_dur_infer = get_phn_dur_list(
                            batch["phn"][0],
                            infer_x_type[0].cpu(),
                            duration_prediction[0].cpu(),
                        )

                        fig = plot.plot_f0(
                            [
                                infer_pitch[0].cpu().numpy(),
                                pitch_prediction[0].cpu().numpy(),
                            ],
                            ["gt full f0", "pred full f0"],
                        )

                        writer.add_figure("pred f0", fig, step)
                        fig = plot.plot_mel_dur(
                            [wav_target_mel, wav_prediction_mel],
                            ["infer gt mel", "infer pred mel"],
                            [phn_dur_train, phn_dur_infer],
                        )
                        infer_wav_target = infer_wav_target[0].cpu()
                        wav_prediction = wav_prediction[0].cpu()
                        writer.add_figure("infer mel", fig, step)
                        writer.add_audio(
                            "infer gt wav",
                            infer_wav_target,
                            step,
                            sample_rate=cfg.dataset.sr,
                        )
                        writer.add_audio(
                            "infer pred audio",
                            wav_prediction,
                            step,
                            sample_rate=cfg.dataset.sr,
                        )
                if step and step % cfg.PortableTTS.train.save_interval == 0:
                    save_path = save_dir / f"{MODEL_NAME}_{step}.pt"
                    torch.save(
                        [
                            i,
                            step,
                            ddp_portable_tts_model.module.state_dict(),
                            ddp_disc_mpd.module.state_dict(),
                            ddp_disc_msd.module.state_dict(),
                            optim_portable_tts.state_dict(),
                            optim_disc.state_dict(),
                            lr_sch_portable_tts.state_dict(),
                            lr_sch_disc.state_dict(),
                        ],
                        save_path,
                    )
            step += 1
        lr_sch_portable_tts.step()
        lr_sch_disc.step()
    # save the model at the end of training
    save_path = save_dir / f"{MODEL_NAME}_{step}.pt"
    torch.save(
        [
            i,
            step,
            ddp_portable_tts_model.module.state_dict(),
            ddp_disc_mpd.module.state_dict(),
            ddp_disc_msd.module.state_dict(),
            optim_portable_tts.state_dict(),
            optim_disc.state_dict(),
            lr_sch_portable_tts.state_dict(),
            lr_sch_disc.state_dict(),
        ],
        save_path,
    )


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



