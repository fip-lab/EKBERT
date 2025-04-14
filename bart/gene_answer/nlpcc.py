import sys
import os
if __name__ == '__main__':
    sys.path.append(os.path.dirname(sys.path[0]))

from transformers import BartForConditionalGeneration, BertTokenizer
from train_code.train_model_nlpcc import bartFineTuner, bartDataset
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

bart_tokenizer = BertTokenizer.from_pretrained('../model_files_chinese')
# t5_model = BartForConditionalGeneration.from_pretrained('../mt5_base_model')
model = torch.load('../bart_trained_model/bart_nlpcc_3.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# t5_model = t5_model.to(device)
model = model.to(device)

nlpcc_kbqa_test_model_gene_answer_address = '../bart_model_gene_answer/nlpcc_kbqa_test_model_gene_answer_3.json'
nlpcc_kbqa_test_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_test_knowques_compansw.json'
with open(nlpcc_kbqa_test_knowques_compansw_address, 'r', encoding='utf-8') as f:
    nlpcc_test_data = json.load(f)
model_answer = []
# for idx, i in enumerate(nlpcc_test_data):
for i in tqdm(nlpcc_test_data):
    # if idx <20:
    item = i[0]
    # item = nlpcc_test_data[0][0]
    test_tokenized = bart_tokenizer.encode_plus(item, return_tensors="pt", truncation=True, max_length=512)
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
        sent = bart_tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        model_answer.append(sent)
    # print(model_answer)
with open(nlpcc_kbqa_test_model_gene_answer_address, 'w', encoding='utf-8') as f:
    json.dump(model_answer, f, ensure_ascii=False)




