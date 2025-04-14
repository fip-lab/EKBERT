#! /bin/bash
device=0
cd ..
cd ..
cd src
cd kbert

CUDA_VISIBLE_DEVICES=${device} python run_kbert_ner.py \
  --pretrained_model_path="../../output/finetune_model/ccks_el_kbert_md/ccks_el_kbert_md.bin" \
  --config_path="../../pretrained_model/bert-base-chinese-google/google_config.json" \
  --vocab_path="../../pretrained_model/bert-base-chinese-google/google_vocab.txt" \
  --train_path="../data/processed/ccks_el_2019/md/train.tsv" \
  --test_path="../data/processed/ccks_el_2019/md/test.tsv" \
  --kg_name="ccks_el_2019" \
  --do_eval \
  --is_test \
  --model_type='ner' \
  --output_model_path="../../output/finetune_model/ccks_el_kbert_md/ccks_el_kbert_md.bin"
