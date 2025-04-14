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
  --train_path="../data/processed/nlpcc_kbqa/ed_kbert/train.tsv" \
  --dev_path="../data/processed/nlpcc_kbqa/ed_kbert/dev.tsv" \
  --epochs_num=1 \
  --batch_size=24 \
  --kg_name="nlpcc_train+nlpcc_test" \
  --do_train \
  --model_type='matching' \
  --output_model_path="../../output/finetune_model/nlpcc_kbqa_kbert/kbert_nlpcc_ed.bin"
