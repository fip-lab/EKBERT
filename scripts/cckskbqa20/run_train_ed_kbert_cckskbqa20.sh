#! /bin/bash
device=0
cd ..
cd ..
cd src
cd kbert

CUDA_VISIBLE_DEVICES=${device} python run_kbert_cls.py \
  --pretrained_model_path="../../pretrained_model/bert-base-chinese-google/google_model.bin" \
  --config_path="../../pretrained_model/bert-base-chinese-google/google_config.json" \
  --vocab_path="../../pretrained_model/bert-base-chinese-google/google_vocab.txt" \
  --train_path="../data/processed/ccks_kbqa_2020/ed_kbert/train.tsv" \
  --dev_path="../data/processed/ccks_kbqa_2020/ed_kbert/dev.tsv" \
  --epochs_num=1 \
  --batch_size=16 \
  --kg_name='ccks_kbqa_2020' \
  --do_train \
  --model_type='matching' \
  --output_model_path="../../output/finetune_model/ccks_kbqa_kbert_ed/ccks_kbqa_kbert_ed.bin"