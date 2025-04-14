#! /bin/bash

device=0

cd ..
cd ..
cd src
cd kbert
# md任务预测
CUDA_VISIBLE_DEVICES=${device} python run_kbert_ner.py \
  --pretrained_model_path="../../output/finetune_model/ccks_el20_kbert_md/ccks_el20_kbert_md.bin" \
  --config_path="../../pretrained_model/bert-base-chinese-google/google_config.json" \
  --vocab_path="../../pretrained_model/bert-base-chinese-google/google_vocab.txt" \
  --train_path="../data/processed/ccks_el_2020/md/train.tsv" \
  --test_path="../data/processed/ccks_el_2020/md/test.tsv" \
  --kg_name="ccks_el_2020" \
  --do_eval \
  --is_test \
  --batch_size=8 \
  --model_type='ner' \
  --output_predict_result_path="../../output/finetune_model/ccks_el20_kbert_md/test_prediction.json" \
  --output_model_path="../../output/finetune_model/ccks_el20_kbert_md/ccks_el20_kbert_md.bin" \
  --do_predict

# # shellcheck disable=SC2103
cd ..
# # # 读md任务预测正确结果，生成筛选后的ed数据
CUDA_VISIBLE_DEVICES=${device} python run_entity_linking.py \
  --model_name='kbert' \
  --load_md_predict_data_path='../output/finetune_model/ccks_el20_kbert_md' \
  --dump_ed_predict_data_path='../src/data/processed/ccks_el_2020/ed_kbert' \
  --dataset='ccksel20'

cd kbert
# ## 读ed数据进行预测，得到el任务结果
CUDA_VISIBLE_DEVICES=${device} python run_kbert_cls.py \
  --pretrained_model_path="../../output/finetune_model/ccks_el20_kbert_ed/kbert_ccksel20_ed.bin" \
  --config_path="../../pretrained_model/bert-base-chinese-google/google_config.json" \
  --vocab_path="../../pretrained_model/bert-base-chinese-google/google_vocab.txt" \
  --train_path="../data/processed/ccks_el_2020/ed_kbert/train.tsv" \
  --test_path="../data/processed/ccks_el_2020/ed_kbert/predict.tsv" \
  --kg_name="ccks_el_2020" \
  --do_eval \
  --is_test \
  --batch_size=8 \
  --model_type='matching' \
  --output_model_path="../../output/finetune_model/ccks_el20_kbert_ed/kbert_ccksel20_ed.bin" \
  --compute_pipeline_el_metric \
  --pipeline_el_md_dir='../../output/finetune_model/ccks_el20_kbert_md' \
  --pipeline_el_ed_dir='../../output/finetune_model/ccks_el20_kbert_ed'
