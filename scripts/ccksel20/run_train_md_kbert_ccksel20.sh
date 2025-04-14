#! /bin/bash
device=0
cd ..
cd ..
cd src
cd kbert

CUDA_VISIBLE_DEVICES=${device} python run_kbert_ner.py \
  --pretrained_model_path="../../pretrained_model/bert-base-chinese-google/google_model.bin" \
  --config_path="../../pretrained_model/bert-base-chinese-google/google_config.json" \
  --vocab_path="../../pretrained_model/bert-base-chinese-google/google_vocab.txt" \
  --train_path="../data/processed/ccks_el_2020/md/train.tsv" \
  --dev_path="../data/processed/ccks_el_2020/md/dev.tsv" \
  --epochs_num=5 \
  --batch_size=8 \
  --do_train \
  --model_type='ner' \
  --kg_name="ccks_el_2020" \
  --output_model_path="../../output/finetune_model/ccks_el20_kbert_md/ccks_el20_kbert_md.bin"
