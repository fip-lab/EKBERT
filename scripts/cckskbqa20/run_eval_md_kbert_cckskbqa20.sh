#! /bin/bash
device=0
cd ..
cd ..
cd src
cd kbert

CUDA_VISIBLE_DEVICES=${device} python run_kbert_ner.py \
  --pretrained_model_path="../../output/finetune_model/ccks_kbqa_kbert_md/ccks_kbqa_kbert_md.bin" \
  --config_path="../../pretrained_model/bert-base-chinese-google/google_config.json" \
  --vocab_path="../../pretrained_model/bert-base-chinese-google/google_vocab.txt" \
  --train_path="../data/processed/ccks_kbqa_2020/md/train.tsv" \
  --test_path="../data/processed/ccks_kbqa_2020/md/test.tsv" \
  --kg_name='ccks_kbqa_2020' \
  --do_eval \
  --is_test \
  --model_type='ner' \
  --batch_size=8 \
  --output_model_path="../../output/finetune_model/ccks_kbqa_kbert_md/ccks_kbqa_kbert_md.bin"
