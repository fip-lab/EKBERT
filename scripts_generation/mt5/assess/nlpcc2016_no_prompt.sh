#! /bin/bash

cd ../../..

cd mt5
cd assess_no_prompt

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