import json
import random

know_ques_answ_address = '../data_no_prompt/nlpcc2018_know_ques_answ_no_prompt.json'


nlpcc2018kbqa_train_knowques_compansw_no_prompt_address = '../data_no_prompt/nlpcc2018_kbqa_train_knowques_compansw_no_prompt.json'
nlpcc2018kbqa_dev_knowques_compansw_no_prompt_address = '../data_no_prompt/nlpcc2018_kbqa_dev_knowques_compansw_no_prompt.json'
nlpcc2018kbqa_test_knowques_compansw_no_prompt_address = '../data_no_prompt/nlpcc2018_kbqa_test_knowques_compansw_no_prompt.json'

with open(know_ques_answ_address, 'r', encoding='utf-8') as f:
    know_ques_answ_no_prompt = json.load(f)

random.seed(42)
random.shuffle(know_ques_answ_no_prompt)

nlpcc2018_kbqa_train_knowques_compansw_no_prompt = know_ques_answ_no_prompt[
                                         :round(0.7 * len(know_ques_answ_no_prompt))]  # round()四舍五入
nlpcc2018_kbqa_dev_knowques_compansw_no_prompt = know_ques_answ_no_prompt[
                                       round(0.7 * len(know_ques_answ_no_prompt)):round(
                                           0.9 * len(know_ques_answ_no_prompt))]
nlpcc2018_kbqa_test_knowques_compansw_no_prompt = know_ques_answ_no_prompt[
                                        round(0.9 * len(know_ques_answ_no_prompt)):]

with open(nlpcc2018kbqa_train_knowques_compansw_no_prompt_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc2018_kbqa_train_knowques_compansw_no_prompt, f, ensure_ascii=False)
with open(nlpcc2018kbqa_dev_knowques_compansw_no_prompt_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc2018_kbqa_dev_knowques_compansw_no_prompt, f, ensure_ascii=False)
with open(nlpcc2018kbqa_test_knowques_compansw_no_prompt_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc2018_kbqa_test_knowques_compansw_no_prompt, f, ensure_ascii=False)

