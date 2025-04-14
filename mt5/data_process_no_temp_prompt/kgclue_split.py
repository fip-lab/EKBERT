import json
import random

know_ques_answ_no_temp_prompt_address = '../data_no_temp_prompt/kgclue_know_ques_answ_no_temp_prompt.json'

kgclue_train_data_no_temp_prompt_address = '../data_no_temp_prompt/kgclue_train_data_no_temp_prompt.json'
kgclue_dev_data_no_temp_prompt_address = '../data_no_temp_prompt/kgclue_dev_data_no_temp_prompt.json'
kgclue_test_data_no_temp_prompt_address = '../data_no_temp_prompt/kgclue_test_data_no_temp_prompt.json'

with open(know_ques_answ_no_temp_prompt_address, 'r', encoding='utf-8') as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)
kgclue_train_data = data[:round(0.7 * len(data))]
kgclue_dev_data = data[round(0.7 * len(data)):round(0.9 * len(data))]
kgclue_test_data = data[round(0.9 * len(data)):]

with open(kgclue_train_data_no_temp_prompt_address, 'w', encoding='utf-8') as f:
    json.dump(kgclue_train_data, f)

with open(kgclue_dev_data_no_temp_prompt_address, 'w', encoding='utf-8') as f:
    json.dump(kgclue_dev_data, f)

with open(kgclue_test_data_no_temp_prompt_address, 'w', encoding='utf-8') as f:
    json.dump(kgclue_test_data, f)

print('kgclue_no_temp_prompt数据处理完毕！')