#! /bin/bash

cd ../../..

cd gpt2
cd assess

cd bleu
python kgclue.py

cd ..
cd meteor
python kgclue.py

cd ..
cd perplexity
python kgclue.py

cd ..
cd rouge
python kgclue.py