import json
import random

kgclue_know_ques_answ_address = '../../data/kgclue_know_ques_answ.json'
kgclue_train_data_address = '../../data/kgclue_train_data.json'
kgclue_dev_data_address = '../../data/kgclue_dev_data.json'
kgclue_test_data_address = '../../data/kgclue_test_data.json'

with open(kgclue_know_ques_answ_address, 'r', encoding='utf-8') as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)
kgclue_train_data = data[:round(0.7 * len(data))]
kgclue_dev_data = data[round(0.7 * len(data)):round(0.9 * len(data))]
kgclue_test_data = data[round(0.9 * len(data)):]

with open(kgclue_train_data_address, 'w', encoding='utf-8') as f:
    json.dump(kgclue_train_data, f)

with open(kgclue_dev_data_address, 'w', encoding='utf-8') as f:
    json.dump(kgclue_dev_data, f)

with open(kgclue_test_data_address, 'w', encoding='utf-8') as f:
    json.dump(kgclue_test_data, f)

print('kgclue数据处理完毕！')