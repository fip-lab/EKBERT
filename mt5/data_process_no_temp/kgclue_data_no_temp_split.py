import json
import random

kgclue_know_ques_answ_no_temp_address = '../data_no_temp/kgclue_know_ques_answ_no_temp.json'
kgclue_train_data_no_temp_address = '../data_no_temp/kgclue_train_data_no_temp.json'
kgclue_dev_data_no_temp_address = '../data_no_temp/kgclue_dev_data_no_temp.json'
kgclue_test_data_no_temp_address = '../data_no_temp/kgclue_test_data_no_temp.json'

with open(kgclue_know_ques_answ_no_temp_address, 'r', encoding='utf-8') as f:
    kgclue_data_no_temp = json.load(f)

random.seed(42)
random.shuffle(kgclue_data_no_temp)
kgclue_train_data = kgclue_data_no_temp[:round(0.7 * len(kgclue_data_no_temp))]
kgclue_dev_data = kgclue_data_no_temp[round(0.7 * len(kgclue_data_no_temp)):round(0.9 * len(kgclue_data_no_temp))]
kgclue_test_data = kgclue_data_no_temp[round(0.9 * len(kgclue_data_no_temp)):]

with open(kgclue_train_data_no_temp_address, 'w', encoding='utf-8') as f:
    json.dump(kgclue_train_data, f)

with open(kgclue_dev_data_no_temp_address, 'w', encoding='utf-8') as f:
    json.dump(kgclue_dev_data, f)

with open(kgclue_test_data_no_temp_address, 'w', encoding='utf-8') as f:
    json.dump(kgclue_test_data, f)

print('kgclue_no_temp数据处理完毕！')