# Copyright (c) 2022 Tsinghua University(Jie Chen)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import pathlib
import os
from typing import Iterable
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, help='Path to BZNSYP dataset')
    parser.add_argument('wav', type=str, help='Path to export paths of wavs.')
    parser.add_argument('speaker', type=str, help='Path to export speakers.')
    parser.add_argument('text', type=str, help='Path to export text of wavs.')
    return parser.parse_args()


def save_scp_files(wav_scp_path: os.PathLike, speaker_scp_path: os.PathLike,
                   text_scp_path: os.PathLike, content: Iterable[str]):
    wav_scp_path = pathlib.Path(wav_scp_path)
    speaker_scp_path = pathlib.Path(speaker_scp_path)
    text_scp_path = pathlib.Path(text_scp_path)

    wav_scp_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_scp_path.parent.mkdir(parents=True, exist_ok=True)
    text_scp_path.parent.mkdir(parents=True, exist_ok=True)

    with open(wav_scp_path, 'w') as wav_scp_file:
        wav_scp_file.writelines([str(line[0]) + '\n' for line in content])
    with open(speaker_scp_path, 'w') as speaker_scp_file:
        speaker_scp_file.writelines([line[1] + '\n' for line in content])
    with open(text_scp_path, 'w') as text_scp_file:
        text_scp_file.writelines([line[2] + '\n' for line in content])


def main(args):
    dataset_dir = pathlib.Path(args.dataset_dir)
    with open(dataset_dir / 'metadata.csv.txt') as train_set_label_file:
        samples = collections.defaultdict(list)
        for line in train_set_label_file:
            sample_name, text, phonemes = line.strip().split('|')
            speaker = 'BZNSYP'

            # edge case
            phonemes = re.sub('P-IY1-guo4\?', 'pi1-guo4?', phonemes)
            phonemes = re.sub('ng1-yuan4-le5', 'en1-yuan4-le5', phonemes)

            # remove other symbols
            phonemes = re.sub('[^a-z0-9\./\-, \?!]', '', phonemes)

            # remove spaces after / and ,
            # substitute space for prosodic word to _
            # substitute - to space
            phonemes = re.sub('/ ', '/', phonemes)
            phonemes = re.sub(', ', ',', phonemes)
            phonemes = re.sub(' ', '_', phonemes)
            phonemes = re.sub('-', ' ', phonemes)

            # add space to split all symbols
            phonemes = re.sub('\?$', ' ?', phonemes)
            phonemes = re.sub('!$', ' !', phonemes)
            phonemes = re.sub('\.$', ' .', phonemes)
            phonemes = re.sub('/', ' / ', phonemes)
            phonemes = re.sub(',', ' , ', phonemes)
            phonemes = re.sub('_', ' _ ', phonemes)

            # remove edge case

            phonemes = re.sub(', \?', '? ', phonemes)
            phonemes = re.sub(', !', '! ', phonemes)
            phonemes = re.sub(', \.', '. ', phonemes)
            phonemes = re.sub('/ \?', '? ', phonemes)
            phonemes = re.sub('/ !', '! ', phonemes)
            phonemes = re.sub('/ \.', '. ', phonemes)

            # here is a strange bug, where Path.exists() will return false
            # for Wave.48k/000001.wav
            wav_path = (dataset_dir / 'Wave.48k' / f'{sample_name}.wav')
            if wav_path.exists():
                samples[speaker].append(
                    (wav_path.absolute(), speaker, phonemes))
    sample_list = []

    for speaker in sorted(samples):
        sample_list.extend(samples[speaker])

    save_scp_files(args.wav, args.speaker, args.text, sample_list)


if __name__ == "__main__":
    main(get_args())
