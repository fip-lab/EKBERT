#! /bin/bash

cd ..
cd ..

cd gpt2
cd train_model
python train_code_nlpcc2018.py

cd ..
cd gene_answer
python nlpcc2018.py

cd ..
cd assess

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