import json
import random

nlpcc2018_know_ques_answ_no_temp_address = '../data_no_temp/nlpcc2018_know_ques_answ_no_temp.json'
nlpcc2018_train_data_no_temp_address = '../data_no_temp/nlpcc2018_train_data_no_temp.json'
nlpcc2018_dev_data_no_temp_address = '../data_no_temp/nlpcc2018_dev_data_no_temp.json'
nlpcc2018_test_data_no_temp_address = '../data_no_temp/nlpcc2018_test_data_no_temp.json'

with open(nlpcc2018_know_ques_answ_no_temp_address, 'r', encoding='utf-8') as f:
    nlpcc_2018_data_no_temp = json.load(f)

random.seed(42)
random.shuffle(nlpcc_2018_data_no_temp)
nlpcc2018_train_data = nlpcc_2018_data_no_temp[:round(0.7 * len(nlpcc_2018_data_no_temp))]
nlpcc2018_dev_data = nlpcc_2018_data_no_temp[round(0.7 * len(nlpcc_2018_data_no_temp)):round(0.9 * len(nlpcc_2018_data_no_temp))]
nlpcc2018_test_data = nlpcc_2018_data_no_temp[round(0.9 * len(nlpcc_2018_data_no_temp)):]

with open(nlpcc2018_train_data_no_temp_address, 'w', encoding='utf-8') as f:
    json.dump(nlpcc2018_train_data, f)

with open(nlpcc2018_dev_data_no_temp_address, 'w', encoding='utf-8') as f:
    json.dump(nlpcc2018_dev_data, f)

with open(nlpcc2018_test_data_no_temp_address, 'w', encoding='utf-8') as f:
    json.dump(nlpcc2018_test_data, f)