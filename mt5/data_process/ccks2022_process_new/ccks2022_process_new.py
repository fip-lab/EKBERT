import json
import re
import random

raw_train_address = '../../src/data/raw/ccks_kbqa_2022/train.txt'

know_ques_answ_address = '../../data/ccks2022_know_ques_answ_new.json'


ccks2022_kbqa_train_knowques_compansw_address = '../../data/ccks2022_kbqa_train_knowques_compansw.json'
ccks2022_kbqa_dev_knowques_compansw_address = '../../data/ccks2022_kbqa_dev_knowques_compansw.json'
ccks2022_kbqa_test_knowques_compansw_address = '../../data/ccks2022_kbqa_test_knowques_compansw.json'

with open(know_ques_answ_address, 'r', encoding='utf-8') as f:
    total_know_ques_compansw = json.load(f)


random.seed(52)
random.shuffle(total_know_ques_compansw)
ccks2022_train_data = total_know_ques_compansw[:round(0.7 * len(total_know_ques_compansw))]
ccks2022_dev_data = total_know_ques_compansw[
               round(0.7 * len(total_know_ques_compansw)):round(0.9 * len(total_know_ques_compansw))]
ccks2022_test_data = total_know_ques_compansw[round(0.9 * len(total_know_ques_compansw)):]

if __name__ == '__main__':
    with open(ccks2022_kbqa_train_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2022_train_data, f, ensure_ascii=False)
    with open(ccks2022_kbqa_dev_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2022_dev_data, f, ensure_ascii=False)
    with open(ccks2022_kbqa_test_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2022_test_data, f, ensure_ascii=False)
