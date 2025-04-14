from transformers import T5Tokenizer, T5ForConditionalGeneration
from mt5.train_code.train_model_ccks2022 import T5FineTuner, mt5Dataset
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

t5_tokenizer = T5Tokenizer.from_pretrained('../mt5_base_model')
t5_model = T5ForConditionalGeneration.from_pretrained('../mt5_base_model')
model = torch.load('../mt5_trained_model/mt5_ccks2022.pkl')
ccks2022_kbqa_test_model_gene_answer_address = '../mt5_model_gene_answer/ccks2022_kbqa_test_model_gene_answer.json'
ccks2022_kbqa_test_knowques_compansw_address = '../data/ccks2022_kbqa_test_knowques_compansw.json'
with open(ccks2022_kbqa_test_knowques_compansw_address, 'r', encoding='utf-8') as f:
    ccks2022_test_data = json.load(f)
model_answer = []
# for idx, i in enumerate(nlpcc_test_data):
for i in tqdm(ccks2022_test_data):
    # if idx <20:
    item = i[0]
    # item = '知识:别名:河边村,中文名称:河边村,所属地区:云南省,下辖地区:昌宁县珠街,面积:1.59平方公里，问题:河边村的占地面积大概有多少?'
    test_tokenized = t5_tokenizer.encode_plus(item, return_tensors="pt")
    test_input_ids = test_tokenized["input_ids"]
    test_attention_mask = test_tokenized["attention_mask"]
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
with open(ccks2022_kbqa_test_model_gene_answer_address, 'w', encoding='utf-8') as f:
    json.dump(model_answer, f, ensure_ascii=False)