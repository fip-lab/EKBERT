import json
import re
import random

raw_train_address = '../../src/data/raw/ccks_kbqa_2022/train.txt'


know_ques_answ_no_temp_address = '../data_no_temp/ccks2022_know_ques_answ_new_no_temp.json'

ccks2022_kbqa_train_knowques_compansw_address = '../data_no_temp/ccks2022_kbqa_train_knowques_compansw.json'
ccks2022_kbqa_dev_knowques_compansw_address = '../data_no_temp/ccks2022_kbqa_dev_knowques_compansw.json'
ccks2022_kbqa_test_knowques_compansw_address = '../data_no_temp/ccks2022_kbqa_test_knowques_compansw.json'

with open(know_ques_answ_no_temp_address, 'r', encoding='utf-8') as f:
    total_know_ques_compansw = json.load(f)


random.seed(42)
random.shuffle(total_know_ques_compansw)
ccks2022_train_data_no_temp = total_know_ques_compansw[:round(0.7 * len(total_know_ques_compansw))]
ccks2022_dev_data_no_temp = total_know_ques_compansw[
               round(0.7 * len(total_know_ques_compansw)):round(0.9 * len(total_know_ques_compansw))]
ccks2022_test_data_no_temp = total_know_ques_compansw[round(0.9 * len(total_know_ques_compansw)):]

if __name__ == '__main__':
    with open(ccks2022_kbqa_train_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2022_train_data_no_temp, f, ensure_ascii=False)
    with open(ccks2022_kbqa_dev_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2022_dev_data_no_temp, f, ensure_ascii=False)
    with open(ccks2022_kbqa_test_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2022_test_data_no_temp, f, ensure_ascii=False)
