#! /bin/bash

cd ..
cd ..

cd mt5
cd train_code_no_prompt
python train_model_nlpcc2018_no_prompt.py

cd ..
cd gene_answer_no_prompt
python nlpcc2018.py

cd ..
cd assess_no_prompt

cd bleu
python nlpcc2018.py

cd ..
cd rouge
python nlpcc2018.py

cd ..
cd meteor
python nlpcc2018.py

cd ..
cd perplexity
python nlpcc2018.py