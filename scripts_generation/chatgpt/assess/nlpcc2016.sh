#! /bin/bash

cd ../../..

cd chatgpt

cd bleu
python nlpcc.py

cd ..
cd meteor
python nlpcc.py

cd ..
cd perplexity
python nlpcc.py

cd ..
cd rouge
python nlpcc.py