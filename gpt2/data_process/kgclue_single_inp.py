import json
from tqdm import tqdm

kgclue_train_data_address = '../../mt5/data/kgclue_train_data.json'
kgclue_dev_data_address = '../../mt5/data/kgclue_dev_data.json'
kgclue_test_data_address = '../../mt5/data/kgclue_test_data.json'

kgclue_kbqa_train_knowques_compansw_new_address = '../data/kgclue_kbqa_train_knowques_compansw_single_inp.json'
kgclue_kbqa_dev_knowques_compansw_new_address = '../data/kgclue_kbqa_dev_knowques_compansw_single_inp.json'
kgclue_kbqa_test_knowques_compansw_new_address = '../data/kgclue_kbqa_test_knowques_compansw_single_inp.json'

with open(kgclue_train_data_address, 'r', encoding='utf-8') as f:
    kgclue_train_data = json.load(f)
with open(kgclue_dev_data_address, 'r', encoding='utf-8') as f:
    kgclue_dev_data = json.load(f)
with open(kgclue_test_data_address, 'r', encoding='utf-8') as f:
    kgclue_test_data = json.load(f)


def create_new_answer_data(data_list):
    # new_answ_data = []
    total_new_list = []
    for item in data_list:
        str_temp = item[0]
        total_new_list.append(str_temp)
    return total_new_list

if __name__ == '__main__':
    kgclue_new_train_answ_data = create_new_answer_data(kgclue_train_data)
    # print(nlpcc_new_answ_data)
    kgclue_new_dev_answ_data = create_new_answer_data(kgclue_dev_data)
    kgclue_new_test_answ_data = create_new_answer_data(kgclue_test_data)
    with open(kgclue_kbqa_train_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(kgclue_new_train_answ_data, f)
    with open(kgclue_kbqa_dev_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(kgclue_new_dev_answ_data, f)
    with open(kgclue_kbqa_test_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(kgclue_new_test_answ_data, f)

print('kgclue_single_inp数据处理完毕！')