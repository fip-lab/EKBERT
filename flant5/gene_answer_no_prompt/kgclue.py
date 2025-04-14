import sys
import os
if __name__ == '__main__':
    sys.path.append(os.path.dirname(sys.path[0]))

from transformers import T5Tokenizer, T5ForConditionalGeneration
from train_code_no_prompt.kgclue import T5FineTuner, mt5Dataset
import torch
import json
import random
import numpy as np
from tqdm import tqdm


# 设置随机种子
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

t5_tokenizer = T5Tokenizer.from_pretrained('../../mt5/mt5_base_model')
t5_model = T5ForConditionalGeneration.from_pretrained('../../mt5/mt5_base_model')
model = torch.load('../flant5_trained_model_no_prompt/mt5_kgclue_no_prompt.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model = t5_model.to(device)
model = model.to(device)

kgclue_kbqa_test_model_gene_answer_address = '../flant5_model_gene_answer_no_prompt/kgclue_kbqa_test_model_gene_answer_no_prompt.json'
kgclue_kbqa_test_knowques_compansw_address = '../../mt5/data_no_prompt/kgclue_test_knowques_compansw_no_prompt.json'
with open(kgclue_kbqa_test_knowques_compansw_address, 'r', encoding='utf-8') as f:
    kgclue_test_data = json.load(f)
model_answer = []
# for idx, i in enumerate(nlpcc_test_data):
for i in tqdm(kgclue_test_data):
    # if idx <20:
    item = i[0]
# item = '知识:别名:河边村,中文名称:河边村,所属地区:云南省,下辖地区:昌宁县珠街,面积:1.59平方公里，问题:河边村的占地面积大概有多少?'
    test_tokenized = t5_tokenizer.encode_plus(item, return_tensors="pt")
    test_input_ids = test_tokenized["input_ids"]
    test_attention_mask = test_tokenized["attention_mask"]
    test_input_ids = test_input_ids.to(device)
    test_attention_mask = test_attention_mask.to(device)

    model.model.eval()
    beam_outputs = model.model.generate(
        input_ids=test_input_ids, attention_mask=test_attention_mask,
        max_length=512,
        early_stopping=True,
        num_beams=10,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=0.8
    )
    # model_answer = []
    for beam_output in beam_outputs:
        sent = t5_tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        model_answer.append(sent)
with open(kgclue_kbqa_test_model_gene_answer_address, 'w', encoding='utf-8') as f:
    json.dump(model_answer, f, ensure_ascii=False)