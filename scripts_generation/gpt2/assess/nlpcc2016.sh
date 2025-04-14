#! /bin/bash

cd ../../..

cd gpt2
cd assess

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