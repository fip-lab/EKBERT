import json
import random

know_ques_answ_address = '../../data/nlpcc2018_know_ques_answ.json'
nlpcc2018_train_data_address = '../../data/nlpcc2018_train_data.json'
nlpcc2018_dev_data_address = '../../data/nlpcc2018_dev_data.json'
nlpcc2018_test_data_address = '../../data/nlpcc2018_test_data.json'

with open(know_ques_answ_address, 'r', encoding='utf-8') as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)
nlpcc2018_train_data = data[:round(0.7 * len(data))]
nlpcc2018_dev_data = data[round(0.7 * len(data)):round(0.9 * len(data))]
nlpcc2018_test_data = data[round(0.9 * len(data)):]

with open(nlpcc2018_train_data_address, 'w', encoding='utf-8') as f:
    json.dump(nlpcc2018_train_data, f)

with open(nlpcc2018_dev_data_address, 'w', encoding='utf-8') as f:
    json.dump(nlpcc2018_dev_data, f)

with open(nlpcc2018_test_data_address, 'w', encoding='utf-8') as f:
    json.dump(nlpcc2018_test_data, f)