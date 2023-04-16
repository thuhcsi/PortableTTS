#!/usr/bin/env bash
# Copyright 2022 Binbin Zhang(binbzha@qq.com), Jie Chen
. path.sh

STAGE=
STOP_STAGE=

DATASET_DIR=./dataset        # path to dataset directory

FASTSPEECH2_DIR=fastspeech2
FASTSPEECH2_FEATURE_DIR=$FASTSPEECH2_DIR/features
PATH_TO_CKPT=

mkdir -p $FASTSPEECH2_FEATURE_DIR

if [ ${STAGE} -le 0 ] && [ ${STOP_STAGE} -ge 0 ]; then
  # Prepare wav.txt, speaker.txt and text.txt
  python local/prepare_data_list.py $DATASET_DIR $FASTSPEECH2_FEATURE_DIR/wav.txt \
    $FASTSPEECH2_FEATURE_DIR/speaker.txt $FASTSPEECH2_FEATURE_DIR/text.txt
  # Prepare special_tokens. Special tokens in BZNSYP are / , _ ? . !
  (echo /; echo ,; echo _; echo ?; echo .; echo !;) > $FASTSPEECH2_FEATURE_DIR/special_token.txt
  # Prepare lexicon.
  python tools/gen_mfa_pinyin_lexicon.py --with-tone --with-r \
    $FASTSPEECH2_FEATURE_DIR/lexicon.txt $FASTSPEECH2_FEATURE_DIR/phone.txt
  # Convert text in text.txt to phonemes.
  python local/convert_text_to_phn.py $FASTSPEECH2_FEATURE_DIR/text.txt \
    $FASTSPEECH2_FEATURE_DIR/lexicon.txt $FASTSPEECH2_FEATURE_DIR/special_token.txt \
    $FASTSPEECH2_FEATURE_DIR/text.txt
fi


if [ ${STAGE} -le 1 ] && [ ${STOP_STAGE} -ge 1 ]; then
  # Prepare alignment lab and pronounciation dictionary for MFA tools
  python local/prepare_alignment.py $FASTSPEECH2_FEATURE_DIR/wav.txt \
    $FASTSPEECH2_FEATURE_DIR/speaker.txt $FASTSPEECH2_FEATURE_DIR/text.txt \
    $FASTSPEECH2_FEATURE_DIR/special_token.txt \
    $FASTSPEECH2_FEATURE_DIR/mfa_pronounciation_dict.txt \
    $FASTSPEECH2_FEATURE_DIR/lab/
fi


if [ ${STAGE} -le 2 ] && [ ${STOP_STAGE} -ge 2 ]; then
  # MFA alignment
  # note that output path for TextGrid should follow multi-speaker dataset
  mfa train -j 16 --phone_set PINYIN --overwrite \
      -a $DATASET_DIR/Wave.48k/ -t $FASTSPEECH2_FEATURE_DIR/mfa_temp \
      $FASTSPEECH2_FEATURE_DIR/lab \
      $FASTSPEECH2_FEATURE_DIR/mfa_pronounciation_dict.txt \
      -o $FASTSPEECH2_FEATURE_DIR/mfa/mfa_model.zip $FASTSPEECH2_FEATURE_DIR/TextGrid/BZNSYP \
      --clean
fi


if [ ${STAGE} -le 3 ] && [ ${STOP_STAGE} -ge 3 ]; then
  python tools/gen_alignment_from_textgrid.py \
    $FASTSPEECH2_FEATURE_DIR/wav.txt \
    $FASTSPEECH2_FEATURE_DIR/speaker.txt $FASTSPEECH2_FEATURE_DIR/text.txt \
    $FASTSPEECH2_FEATURE_DIR/special_token.txt $FASTSPEECH2_FEATURE_DIR/TextGrid \
    $FASTSPEECH2_FEATURE_DIR/aligned_wav.txt \
    $FASTSPEECH2_FEATURE_DIR/aligned_speaker.txt \
    $FASTSPEECH2_FEATURE_DIR/duration.txt \
    $FASTSPEECH2_FEATURE_DIR/aligned_text.txt
  # speaker to id map
  cat $FASTSPEECH2_FEATURE_DIR/aligned_speaker.txt | awk '{print $1}' | sort | uniq | \
      awk '{print $1, NR-1}' > $FASTSPEECH2_FEATURE_DIR/spk2id
  # phone to id map
  python tools/gen_phn2id.py $FASTSPEECH2_FEATURE_DIR/lexicon.txt \
    $FASTSPEECH2_FEATURE_DIR/special_token.txt $FASTSPEECH2_FEATURE_DIR/phn2id
fi


if [ ${STAGE} -le 4 ] && [ ${STOP_STAGE} -ge 4 ]; then
  # generate training, validation and test samples
  python local/train_val_test_split.py $FASTSPEECH2_FEATURE_DIR/aligned_wav.txt \
  $FASTSPEECH2_FEATURE_DIR/aligned_speaker.txt \
  $FASTSPEECH2_FEATURE_DIR/aligned_text.txt \
  $FASTSPEECH2_FEATURE_DIR/duration.txt $FASTSPEECH2_FEATURE_DIR
fi


if [ ${STAGE} -le 5 ] && [ ${STOP_STAGE} -ge 5 ]; then
  # Prepare training samples
  python local/make_data_list.py $FASTSPEECH2_FEATURE_DIR/train/train_wav.txt \
      $FASTSPEECH2_FEATURE_DIR/train/train_speaker.txt \
      $FASTSPEECH2_FEATURE_DIR/train/train_text.txt \
      $FASTSPEECH2_FEATURE_DIR/train/train_duration.txt \
      $FASTSPEECH2_FEATURE_DIR/train/datalist.jsonl
  # Prepare validation samples
  python local/make_data_list.py $FASTSPEECH2_FEATURE_DIR/val/val_wav.txt \
      $FASTSPEECH2_FEATURE_DIR/val/val_speaker.txt \
      $FASTSPEECH2_FEATURE_DIR/val/val_text.txt \
      $FASTSPEECH2_FEATURE_DIR/val/val_duration.txt \
      $FASTSPEECH2_FEATURE_DIR/val/datalist.jsonl
  # Prepare test samples
  python local/make_data_list.py $FASTSPEECH2_FEATURE_DIR/test/test_wav.txt \
      $FASTSPEECH2_FEATURE_DIR/test/test_speaker.txt \
      $FASTSPEECH2_FEATURE_DIR/test/test_text.txt \
      $FASTSPEECH2_FEATURE_DIR/test/test_duration.txt \
      $FASTSPEECH2_FEATURE_DIR/test/datalist.jsonl
fi


if [ ${STAGE} -le 6 ] && [ ${STOP_STAGE} -ge 6 ]; then
  # Train Portable TTS
  python portable_tts/train.py -c conf/portable_tts.yaml -p 12349
fi


if [ ${STAGE} -le 7 ] && [ ${STOP_STAGE} -ge 7 ]; then
  # Inference Portable TTS
  python portable_tts/portable_tts/inference.py --config conf/portable_tts.yaml \
    --ckpt_path $PATH_TO_CKPT --datalist_path $FASTSPEECH2_FEATURE_DIR/test/datalist.jsonl \
    --output_path output_wav
fi
