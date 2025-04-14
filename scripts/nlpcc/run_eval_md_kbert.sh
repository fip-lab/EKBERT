#! /bin/bash
device=0
cd ..
cd ..
cd src
cd kbert

CUDA_VISIBLE_DEVICES=${device} python run_kbert_ner.py \
  --pretrained_model_path="../../output/finetune_model/nlpcc_kbqa_kbert_md/kbert_nlpcc_md.bin" \
  --config_path="../../pretrained_model/bert-base-chinese-google/google_config.json" \
  --vocab_path="../../pretrained_model/bert-base-chinese-google/google_vocab.txt" \
  --train_path="../data/processed/nlpcc_kbqa/md/train.tsv" \
  --test_path="../data/processed/nlpcc_kbqa/md/test.tsv" \
  --kg_name="nlpcc" \
  --do_eval \
  --is_test \
  --batch_size=24 \
  --model_type='ner' \
  --output_model_path="../../output/finetune_model/nlpcc_kbqa_kbert_md/kbert_nlpcc_md.bin"
