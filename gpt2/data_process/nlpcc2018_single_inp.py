import json
from tqdm import tqdm

nlpcc2018_train_data_address = '../../mt5/data/nlpcc2018_train_data.json'
nlpcc2018_dev_data_address = '../../mt5/data/nlpcc2018_dev_data.json'
nlpcc2018_test_data_address = '../../mt5/data/nlpcc2018_test_data.json'

nlpcc2018_kbqa_train_knowques_compansw_new_address = '../data/nlpcc2018_kbqa_train_knowques_compansw_single_inp.json'
nlpcc2018_kbqa_dev_knowques_compansw_new_address = '../data/nlpcc2018_kbqa_dev_knowques_compansw_single_inp.json'
nlpcc2018_kbqa_test_knowques_compansw_new_address = '../data/nlpcc2018_kbqa_test_knowques_compansw_single_inp.json'

with open(nlpcc2018_train_data_address, 'r', encoding='utf-8') as f:
    nlpcc2018_train_data = json.load(f)
with open(nlpcc2018_dev_data_address, 'r', encoding='utf-8') as f:
    nlpcc2018_dev_data = json.load(f)
with open(nlpcc2018_test_data_address, 'r', encoding='utf-8') as f:
    nlpcc2018_test_data = json.load(f)


def create_new_answer_data(data_list):
    # new_answ_data = []
    total_new_list = []
    for item in data_list:
        str_temp = item[0]
        total_new_list.append(str_temp)
    return total_new_list

if __name__ == '__main__':
    nlpcc2018_new_train_answ_data = create_new_answer_data(nlpcc2018_train_data)
    # print(nlpcc_new_answ_data)
    nlpcc2018_new_dev_answ_data = create_new_answer_data(nlpcc2018_dev_data)
    nlpcc2018_new_test_answ_data = create_new_answer_data(nlpcc2018_test_data)
    with open(nlpcc2018_kbqa_train_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc2018_new_train_answ_data, f)
    with open(nlpcc2018_kbqa_dev_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc2018_new_dev_answ_data, f)
    with open(nlpcc2018_kbqa_test_knowques_compansw_new_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc2018_new_test_answ_data, f)