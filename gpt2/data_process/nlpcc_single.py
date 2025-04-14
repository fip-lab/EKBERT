import json
from tqdm import tqdm

nlpcc_kbqa_train_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_train_knowques_compansw.json'
nlpcc_kbqa_dev_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_dev_knowques_compansw.json'
nlpcc_kbqa_test_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_test_knowques_compansw.json'

nlpcc_kbqa_train_knowques_compansw_new_address = '../data/nlpcc_kbqa_train_knowques_compansw_single.json'
nlpcc_kbqa_dev_knowques_compansw_new_address = '../data/nlpcc_kbqa_dev_knowques_compansw_single.json'
nlpcc_kbqa_test_knowques_compansw_new_address = '../data/nlpcc_kbqa_test_knowques_compansw_single.json'

with open(nlpcc_kbqa_train_knowques_compansw_address, 'r', encoding='utf-8') as f:
    nlpcc_train_data = json.load(f)
with open(nlpcc_kbqa_dev_knowques_compansw_address, 'r', encoding='utf-8') as f:
    nlpcc_dev_data = json.load(f)
with open(nlpcc_kbqa_test_knowques_compansw_address, 'r', encoding='utf-8') as f:
    nlpcc_test_data = json.load(f)


def create_new_answer_data(data_list):
    # new_answ_data = []
    total_new_list = []
    for item in data_list:
        str_temp = item[0] + item[1]
        total_new_list.append(str_temp)
    return total_new_list

if __name__ == '__main__':
    nlpcc_new_train_answ_data = create_new_answer_data(nlpcc_train_data)
    # print(nlpcc_new_answ_data)
    nlpcc_new_dev_answ_data = create_new_answer_data(nlpcc_dev_data)
    nlpcc_new_test_answ_data = create_new_answer_data(nlpcc_test_data)
    with open(nlpcc_kbqa_train_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc_new_train_answ_data, f)
    with open(nlpcc_kbqa_dev_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc_new_dev_answ_data, f)
    with open(nlpcc_kbqa_test_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc_new_test_answ_data, f)

print('nlpcc_single数据处理完毕！')