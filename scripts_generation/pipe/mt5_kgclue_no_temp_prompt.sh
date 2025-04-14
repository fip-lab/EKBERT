#! /bin/bash

cd ..
cd ..

cd mt5
cd data_process_no_temp_prompt
python kgclue.py
python kgclue_split.py

cd ..
cd train_code_no_temp_prompt
python train_model_kgclue_no_temp_prompt.py

cd ..
cd gene_answer_no_temp_prompt
python kgclue.py

cd ..
cd assess_no_temp_prompt

cd bleu
python kgclue.py

cd ..
cd rouge
python kgclue.py

cd ..
cd meteor
python kgclue.py

cd ..
cd perplexity
python kgclue.py