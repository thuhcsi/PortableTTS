# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import numpy as np
import torch
import librosa
import jsonlines
from torch.nn.utils import rnn

from feature import spectrogram, pitch


class PortableTTSDataset(torch.utils.data.Dataset):

    def __init__(self, datalist_path, phn2id_path, spk2id_path,
                 special_token_path, sr, preemphasis, n_fft, win_size,
                 hop_size, n_mel, mel_f_min, min_level_db, ref_level_db,
                 pitch_min, pitch_max) -> None:
        self.datalist_path = datalist_path
        self.phn2id_path = phn2id_path
        self.spk2id_path = spk2id_path
        self.special_token_path = special_token_path
        self.sr = sr
        self.hop_size = hop_size
        self.phn2id = None
        self.spk2id = None
        self.sty2id = None
        self.special_token = None
        self.datalist = None
        self.spectrogram_extractor = spectrogram.SpectrogramExtractor(
            sr, preemphasis, n_fft, win_size, hop_size, n_mel, mel_f_min,
            min_level_db, ref_level_db)
        self.pitch_extractor = pitch.Pitch(sr, hop_size, pitch_min, pitch_max)

    def to_id(self, seq, seq2id):
        return [seq2id[x] for x in seq]

    def get_duration(self, duration):
        # phonemes should last at least one frame
        return [
            round(end * self.sr / self.hop_size) -
            round(start * self.sr / self.hop_size) for start, end in duration
        ]

    def get_token_type(self, phn):
        token_type = []
        for x in phn:
            if x in self.special_token:
                token_type.append(0)
            else:
                token_type.append(1)
        return token_type

    def __getitem__(self, index):
        self._load()
        sample = self.datalist[index]

        audio_path = sample['wav_path']
        spk = sample['speaker']
        phn = sample['text']
        duration = sample['duration']
        sample_id = sample['key']

        duration = [0]+torch.tensor(sample['duration']).cumsum(dim=0).tolist()
        duration = self.get_duration(zip(duration[:-1], duration[1:]))
        audio, spec = self.get_audio(audio_path)
        phn_idx = self.to_id(phn, self.phn2id)
        #spk_idx = self.to_id(spk, self.spk2id)
        token_type = self.get_token_type(phn)
        f0 = torch.from_numpy(
            self.pitch_extractor.get_pitch(audio,
                                           use_token_averaged_pitch=True,
                                           duration=np.array(duration))).float()  # (t)

        
        # align duration with spectrogram
        diff = spec.size(0) - sum(duration)
        if diff != 0:
            if diff > 0:
                duration[-1] += diff
            elif duration[-1] + diff > 0:
                duration[-1] += diff
            elif duration[0] + diff > 0:
                duration[0] += diff
            else:
                raise ValueError(
                    f'Duration and spec mismatch: {sample_id} with {diff}!')
        assert sum(duration) == spec.size(0)
        return {
            'sample_id': sample_id,
            'phn': phn,
            'phn_idx': phn_idx,
            'spk': spk,
            #'spk_idx': spk_idx,
            'duration': duration,
            'type': token_type,
            'spectrogram': spec,  # (t,d)
            'audio': audio,
            'f0': f0  # (t)
        }

    def __len__(self):
        self._load()
        return len(self.datalist)

    def _load(self):
        self._load_datalist()
        self._load_phn2id()
        self._load_spk2id()
        self._load_special_tokens()

    def _load_phn2id(self):
        if self.phn2id is None:
            self.phn2id = {}
            with open(self.phn2id_path) as f:
                for i, line in enumerate(f):
                    # zero is reserved for padding
                    phn, _ = line.strip().split(' ')
                    self.phn2id[phn] = i

    def _load_spk2id(self):
        if self.spk2id is None:
            self.spk2id = {}
            with open(self.spk2id_path) as f:
                for line in f:
                    spk, id = line.strip().split(' ')
                    self.spk2id[spk] = int(id)

    def _load_special_tokens(self):
        if self.special_token is None:
            self.special_token = set()
            with open(self.special_token_path) as f:
                for line in f:
                    self.special_token.add(line.strip())

    def _load_datalist(self):
        if self.datalist is None:
            with jsonlines.open(self.datalist_path) as f:
                self.datalist = [x for x in f]

    def get_audio(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.sr)
        audio = torch.from_numpy(audio)
        spec = self.spectrogram_extractor(audio, 'mel')
        return audio, spec


class PortableTTSCollateFn:

    def __call__(self, batch):
        """
        batch: list of {phn,phn_idx,sty,sty_idx,spk,spk_idx,duration,type,face,
        spec,audio}
        """
        batch_phn_length, batch_idx = torch.sort(torch.tensor(
            [len(sample['phn_idx']) for sample in batch], dtype=torch.long),
                                                 descending=True)
        batch_sample_id = [batch[i]['sample_id'] for i in batch_idx]
        batch_phn = [batch[i]['phn'] for i in batch_idx]
        batch_phn_idx = rnn.pad_sequence([
            torch.tensor(batch[i]['phn_idx'], dtype=torch.long)
            for i in batch_idx
        ],
                                         batch_first=True)  # (b,t,d)
        #batch_spk_id = torch.tensor([batch[i]['spk_idx'] for i in batch_idx],
        #                            dtype=torch.long)
        batch_duration = rnn.pad_sequence(
            [torch.tensor(batch[i]['duration']) for i in batch_idx],
            batch_first=True)  # (b,t)
        batch_duration_length = torch.tensor(
            [len(batch[i]['duration']) for i in batch_idx], dtype=torch.long)

        batch_type = rnn.pad_sequence([
            torch.tensor(batch[i]['type'], dtype=torch.long) for i in batch_idx
        ],
                                      batch_first=True)  # (b,t)
        batch_spec = rnn.pad_sequence(
            [batch[i]['spectrogram'] for i in batch_idx],
            batch_first=True).permute(0, 2, 1)  # (b,d,t)
        batch_audio = rnn.pad_sequence([batch[i]['audio'] for i in batch_idx],
                                       batch_first=True)  # (b,d,t)
        # assume face and mel have the same length
        batch_spec_length = torch.tensor(
            [batch[i]['spectrogram'].size(0) for i in batch_idx],
            dtype=torch.long)  # (b)

        batch_f0 = rnn.pad_sequence([batch[i]['f0'] for i in batch_idx],
                                    batch_first=True)  # (b,t)

        return {
            'sample_id': batch_sample_id,
            'phn': batch_phn,
            'phn_idx': batch_phn_idx,
            'phn_length': batch_phn_length,
            'duration': batch_duration,
            'duration_length': batch_duration_length,
            'type': batch_type,
            #'face': batch_face,
            'spectrogram': batch_spec,
            'audio': batch_audio,
            'spectrogram_length': batch_spec_length,
            'f0': batch_f0
        }
