#! /bin/bash

cd ..
cd ..

cd mt5
cd data_process_no_prompt
python nlpcc_process.py

cd ..
cd train_code_no_prompt
python train_model_nlpcc_no_prompt.py

cd ..
cd gene_answer_no_prompt
python nlpcc.py

cd ..
cd assess_no_prompt

cd bleu
python nlpcc.py

cd ..
cd rouge
python nlpcc.py

cd ..
cd meteor
python nlpcc.py

cd ..
cd perplexity
python nlpcc.py