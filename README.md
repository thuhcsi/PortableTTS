# PortableTTS: Lightweight End-to-end Text-to-Speech
PortableTTS is based on [PortaSpeech](https://github.com/NATSpeech/NATSpeech) and [iSTFTNet](https://github.com/rishikksh20/iSTFTNet-pytorch).

The text encoder and decoder are based on [FFT blocks](https://arxiv.org/abs/1905.09263).

Then, decoder output is converted to wave by iSTFTNet-C8C8I.

Demos are available at: https://thuhcsi.github.io/PortableTTS/
## Setup environment

Install [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html).

Then, run:
```bash
python -m pip install -r requirements.txt
```

## Prepare dataset for BZNSYP

Edit `BZNSYP/run.sh`, set `STAGE` and `STOP_STAGE` to `0` and `5`, respectively.
Set `DATASET_DIR` to directory containing `metadata.csv.txt`.

Then, run:
```bash
bash run.sh
```
Note: should use python environment where Montreal Forced Aligner is installed.

## Train BZNSYP

Edit `BZNSYP/conf/portable_tts.yaml`, set `datalist_path`, `phn2id_path`, 
`special_tokens_path` and `spk2id_path` to paths of corresponding files generated 
above.

Edit `BZNSYP/run.sh`, set `STAGE` and `STOP_STAGE` to `6` and `6`, respectively.

Then, run:
```bash
bash run.sh
```


## Preprocess BZNSYP
Edit `BZNSYP/run.sh`, set `STAGE` and `STOP_STAGE` to `7` and `7`, respectively.
Also, set `PATH_TO_CKPT` to model checkpoint.

Then, run:
```bash
bash run.sh
```
and this will generate inference samples in `datalist.jsonl` of test set.

## References

Preprocess scripts from [wetts](https://github.com/wenet-e2e/wetts).

Encoder and decoder from [PortaSpeech](https://github.com/NATSpeech/NATSpeech).

Vocoder from [iSTFTNet](https://github.com/rishikksh20/iSTFTNet-pytorch).

Architecture from [JETS](https://arxiv.org/abs/2203.16852).