#! /bin/bash

cd ..
cd ..

cd mt5
cd data_process_no_temp
python nlpcc_process.py

cd ..
cd train_code_no_temp
python train_model_nlpcc_no_temp.py

cd ..
cd gene_answer_no_temp
python nlpcc.py

cd ..
cd assess_no_temp

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