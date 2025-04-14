#! /bin/bash
device=0
cd ..
cd ..
cd src
cd kbert

CUDA_VISIBLE_DEVICES=${device} python run_kbert_cls.py \
  --pretrained_model_path="../../output/finetune_model/ccks_el20_kbert_ed/kbert_ccksel20_ed.bin" \
  --config_path="../../pretrained_model/bert-base-chinese-google/google_config.json" \
  --vocab_path="../../pretrained_model/bert-base-chinese-google/google_vocab.txt" \
  --train_path="../data/processed/ccks_el_2020/ed_kbert/train.tsv" \
  --test_path="../data/processed/ccks_el_2020/ed_kbert/test.tsv" \
  --kg_name="ccks_el_2020" \
  --do_eval \
  --is_test \
  --model_type='matching' \
  --output_model_path="../../output/finetune_model/ccks_el20_kbert_ed/kbert_ccksel20_ed.bin"
