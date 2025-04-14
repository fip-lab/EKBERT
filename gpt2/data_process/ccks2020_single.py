import json
from tqdm import tqdm

ccks2020_kbqa_train_knowques_compansw_address = '../../mt5/data/ccks2020_kbqa_train_knowques_compansw.json'
ccks2020_kbqa_dev_knowques_compansw_address = '../../mt5/data/ccks2020_kbqa_dev_knowques_compansw.json'
ccks2020_kbqa_test_knowques_compansw_address = '../../mt5/data/ccks2020_kbqa_test_knowques_compansw.json'

ccks2020_kbqa_train_knowques_compansw_new_address = '../data/ccks2020_kbqa_train_knowques_compansw_single.json'
ccks2020_kbqa_dev_knowques_compansw_new_address = '../data/ccks2020_kbqa_dev_knowques_compansw_single.json'
ccks2020_kbqa_test_knowques_compansw_new_address = '../data/ccks2020_kbqa_test_knowques_compansw_single.json'

with open(ccks2020_kbqa_train_knowques_compansw_address, 'r', encoding='utf-8') as f:
    ccks2020_train_data = json.load(f)
with open(ccks2020_kbqa_dev_knowques_compansw_address, 'r', encoding='utf-8') as f:
    ccks2020_dev_data = json.load(f)
with open(ccks2020_kbqa_test_knowques_compansw_address, 'r', encoding='utf-8') as f:
    ccks2020_test_data = json.load(f)


def create_new_answer_data(data_list):
    # new_answ_data = []
    total_new_list = []
    for item in data_list:
        str_temp = item[0] + item[1]
        # if len(str_temp)<=512:
        total_new_list.append(str_temp)
    return total_new_list

if __name__ == '__main__':
    ccks2020_new_train_answ_data = create_new_answer_data(ccks2020_train_data)
    # print(nlpcc_new_answ_data)
    ccks2020_new_dev_answ_data = create_new_answer_data(ccks2020_dev_data)
    ccks2020_new_test_answ_data = create_new_answer_data(ccks2020_test_data)
    with open(ccks2020_kbqa_train_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2020_new_train_answ_data, f)
    with open(ccks2020_kbqa_dev_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2020_new_dev_answ_data, f)
    with open(ccks2020_kbqa_test_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2020_new_test_answ_data, f)