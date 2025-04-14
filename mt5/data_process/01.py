import json
import re
import random
from tqdm import tqdm


# nlpcc_kbqa_train_knowques_compansw_address = '../data/nlpcc_kbqa_train_knowques_compansw.json'
# nlpcc_kbqa_dev_knowques_compansw_address = '../data/nlpcc_kbqa_dev_knowques_compansw.json'
# nlpcc_kbqa_test_knowques_compansw_address = '../data/nlpcc_kbqa_test_knowques_compansw.json'

kgclue_train_data_address = '../data/kgclue_train_data.json'
kgclue_dev_data_address = '../data/kgclue_dev_data.json'
kgclue_test_data_address = '../data/kgclue_test_data.json'


# nlpcc_kbqa_train_knowques_compansw_address = '../data_no_prompt/nlpcc_kbqa_train_knowques_compansw.json'
# nlpcc_kbqa_dev_knowques_compansw_address = '../data_no_prompt/nlpcc_kbqa_dev_knowques_compansw.json'
# nlpcc_kbqa_test_knowques_compansw_address = '../data_no_prompt/nlpcc_kbqa_test_knowques_compansw.json'

kgclue_train_knowques_compansw_no_prompt_address = '../data_no_prompt/kgclue_train_knowques_compansw_no_prompt.json'
kgclue_dev_knowques_compansw_no_prompt_address = '../data_no_prompt/kgclue_dev_knowques_compansw_no_prompt.json'
kgclue_test_knowques_compansw_no_prompt_address = '../data_no_prompt/kgclue_test_knowques_compansw_no_prompt.json'


# nlpcc_kbqa_train_knowques_compansw_address = '../data_no_temp/nlpcc_kbqa_train_knowques_compansw.json'
# nlpcc_kbqa_dev_knowques_compansw_address = '../data_no_temp/nlpcc_kbqa_dev_knowques_compansw.json'
# nlpcc_kbqa_test_knowques_compansw_address = '../data_no_temp/nlpcc_kbqa_test_knowques_compansw.json'

kgclue_train_data_no_temp_address = '../data_no_temp/kgclue_train_data_no_temp.json'
kgclue_dev_data_no_temp_address = '../data_no_temp/kgclue_dev_data_no_temp.json'
kgclue_test_data_no_temp_address = '../data_no_temp/kgclue_test_data_no_temp.json'


nlpcc_kbqa_train_knowques_compansw_address = '../data_no_temp_prompt/nlpcc_kbqa_train_knowques_compansw.json'
nlpcc_kbqa_dev_knowques_compansw_address = '../data_no_temp_prompt/nlpcc_kbqa_dev_knowques_compansw.json'
nlpcc_kbqa_test_knowques_compansw_address = '../data_no_temp_prompt/nlpcc_kbqa_test_knowques_compansw.json'

kgclue_train_data_no_temp_prompt_address = '../data_no_temp_prompt/kgclue_train_data_no_temp_prompt.json'
kgclue_dev_data_no_temp_prompt_address = '../data_no_temp_prompt/kgclue_dev_data_no_temp_prompt.json'
kgclue_test_data_no_temp_prompt_address = '../data_no_temp_prompt/kgclue_test_data_no_temp_prompt.json'

nlpcc_kbqa_test_model_gene_answer_no_temp_address = '../mt5_model_gene_answer_no_temp/nlpcc_kbqa_test_model_gene_answer_no_temp.json'

kgclue_kbqa_test_model_gene_answer_no_temp_address = '../mt5_model_gene_answer_no_temp/kgclue_kbqa_test_model_gene_answer_no_temp.json'

with open(kgclue_kbqa_test_model_gene_answer_no_temp_address, 'r') as f:
    p = json.load(f)

# with open(kgclue_dev_data_address, 'r') as f:
#     p2 = json.load(f)
#
# with open(kgclue_test_data_address, 'r') as f:
#     p3 = json.load(f)

for idx,item in enumerate(p):
    if idx<500:
        print(item)


# len1 = 0
# for item in p1:
#     len1 += len(item[0])
# for item in p2:
#     len1 += len(item[0])
# for item in p3:
#     len1 += len(item[0])
# avg = len1/(len(p1)+len(p2)+len(p3))
# print(avg)