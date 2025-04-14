#! /bin/bash

cd ..
cd ..

cd mt5
cd data_process_no_prompt
python kgclue_process.py
python kgclue_process_split.py

cd ..
cd train_code_no_prompt
python train_model_kgclue_no_prompt.py

cd ..
cd gene_answer_no_prompt
python kgclue.py

cd ..
cd assess_no_prompt

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