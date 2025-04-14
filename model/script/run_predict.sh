#! /bin/bash

device=0
cd ..
cd model

# ed任务
ed_train_name="finetune_model/$1"
ed_model_dir="finetune_model/$1"
ed_data_dir="data"

# 读ed数据进行预测，得到el任务结果
CUDA_VISIBLE_DEVICES=${device} python main_savebestacc.py \
  --model_type=bert_seq_cls \
  --task_name=sfarerank \
  --data_dir=${ed_data_dir} \
  --model_name_or_path="${ed_model_dir}" \
  --output_dir="${ed_train_name}" \
  --max_seq_length=60 \
  --do_eval \
  --per_gpu_eval_batch_size=100 \
  --input_test_name='predict-1128.tsv'
