#!/usr/bin/env python3
# Copyright (c) 2022 Jie Chen
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

from wetts.utils import constants


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('lexicon', type=str, help='Path to lexicon.txt.')
    parser.add_argument('special_tokens',
                        type=str,
                        help='Path to special_token.txt.')
    parser.add_argument('output', type=str, help='Path to phn2id')
    return parser.parse_args()


def main(args):
    phns = []
    with open(args.lexicon) as flexicon:
        for line in flexicon:
            tokens = line.strip().split()
            for phoneme in tokens[1:]:
                phns.append(phoneme)
    with open(args.special_tokens) as fsp:
        for line in fsp:
            phns.append(line.strip())
    phns = [constants.PAD_TOKEN] + sorted(
        list(set(phns) | constants.SPECIAL_PHONES))
    with open(args.output, 'w') as foutput:
        for i, phn in enumerate(phns):
            foutput.write(f'{phn} {i}\n')


if __name__ == '__main__':
    main(get_args())
