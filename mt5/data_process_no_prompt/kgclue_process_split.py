import json
import random

know_ques_answ_no_prompt_address = '../data_no_prompt/kgclue_know_ques_answ_no_prompt.json'


kgclue_train_knowques_compansw_no_prompt_address = '../data_no_prompt/kgclue_train_knowques_compansw_no_prompt.json'
kgclue_dev_knowques_compansw_no_prompt_address = '../data_no_prompt/kgclue_dev_knowques_compansw_no_prompt.json'
kgclue_test_knowques_compansw_no_prompt_address = '../data_no_prompt/kgclue_test_knowques_compansw_no_prompt.json'

with open(know_ques_answ_no_prompt_address, 'r', encoding='utf-8') as f:
    know_ques_answ_no_prompt = json.load(f)

random.seed(42)
random.shuffle(know_ques_answ_no_prompt)

kgclue_train_knowques_compansw_no_prompt = know_ques_answ_no_prompt[
                                         :round(0.7 * len(know_ques_answ_no_prompt))]  # round()四舍五入
kgclue_dev_knowques_compansw_no_prompt = know_ques_answ_no_prompt[
                                       round(0.7 * len(know_ques_answ_no_prompt)):round(
                                           0.9 * len(know_ques_answ_no_prompt))]
kgclue_test_knowques_compansw_no_prompt = know_ques_answ_no_prompt[
                                        round(0.9 * len(know_ques_answ_no_prompt)):]

with open(kgclue_train_knowques_compansw_no_prompt_address, 'w', encoding='utf-8') as f:
        json.dump(kgclue_train_knowques_compansw_no_prompt, f, ensure_ascii=False)
with open(kgclue_dev_knowques_compansw_no_prompt_address, 'w', encoding='utf-8') as f:
        json.dump(kgclue_dev_knowques_compansw_no_prompt, f, ensure_ascii=False)
with open(kgclue_test_knowques_compansw_no_prompt_address, 'w', encoding='utf-8') as f:
        json.dump(kgclue_test_knowques_compansw_no_prompt, f, ensure_ascii=False)

print('kgclue_no_prompt数据处理完毕！')