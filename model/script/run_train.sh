#! /bin/bash


train_name="finetune_model/$1"
model_dir="../pretrained_model/bert-base-chinese-pytorch"
data_dir="data"
device=0
cd ..
# cd model

CUDA_VISIBLE_DEVICES=${device} python main_savebestacc.py \
  --model_type=bert_seq_cls \
  --task_name=sfarerank \
  --data_dir=${data_dir} \
  --model_name_or_path=${model_dir} \
  --output_dir="${train_name}" \
  --max_seq_length=60 \
  --do_train \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=100 \
  --learning_rate=5e-5 \
  --weight_decay=1e-2 \
  --warmup_steps=500 \
  --num_train_epochs=5 \
  --overwrite_output_dir \
  --evaluate_during_training \
  --input_test_name=dev.tsv
