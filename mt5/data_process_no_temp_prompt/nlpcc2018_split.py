import json
import random

know_ques_answ_no_temp_prompt_address = '../data_no_temp_prompt/nlpcc2018_know_ques_answ_no_temp_prompt.json'
nlpcc2018_train_data_no_temp_prompt_address = '../data_no_temp_prompt/nlpcc2018_train_data_no_temp_prompt.json'
nlpcc2018_dev_data_no_temp_prompt_address = '../data_no_temp_prompt/nlpcc2018_dev_data_no_temp_prompt.json'
nlpcc2018_test_data_no_temp_prompt_address = '../data_no_temp_prompt/nlpcc2018_test_data_no_temp_prompt.json'

with open(know_ques_answ_no_temp_prompt_address, 'r', encoding='utf-8') as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)
nlpcc2018_train_data = data[:round(0.7 * len(data))]
nlpcc2018_dev_data = data[round(0.7 * len(data)):round(0.9 * len(data))]
nlpcc2018_test_data = data[round(0.9 * len(data)):]

with open(nlpcc2018_train_data_no_temp_prompt_address, 'w', encoding='utf-8') as f:
    json.dump(nlpcc2018_train_data, f)

with open(nlpcc2018_dev_data_no_temp_prompt_address, 'w', encoding='utf-8') as f:
    json.dump(nlpcc2018_dev_data, f)

with open(nlpcc2018_test_data_no_temp_prompt_address, 'w', encoding='utf-8') as f:
    json.dump(nlpcc2018_test_data, f)