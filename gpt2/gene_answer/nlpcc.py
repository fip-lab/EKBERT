from transformers import GPT2LMHeadModel, BertTokenizer, pipeline
import json
import torch
import random
from tqdm import tqdm

# 设置随机种子
seed = 42
torch.manual_seed(seed)
random.seed(seed)

nlpcc_kbqa_test_knowques_compansw_new_inp_address = '../data/nlpcc_kbqa_test_knowques_compansw_single_inp.json'
nlpcc_kbqa_test_model_gene_answer_address = '../gpt2_model_gene_answer/nlpcc_kbqa_test_model_gene_answer.json'

with open(nlpcc_kbqa_test_knowques_compansw_new_inp_address, 'r', encoding='utf-8') as f:
    nlpcc_kbqa_test_knowques_compansw_new_inp = json.load(f)
# 加载训练好的模型和分词器
tokenizer = BertTokenizer.from_pretrained('../gpt2_trained_model/gpt2_trained_model_nlpcc')
model = GPT2LMHeadModel.from_pretrained('../gpt2_trained_model/gpt2_trained_model_nlpcc')

# 读取模型和tokenizer
# tokenizer = AutoTokenizer.from_pretrained('模型路径')
# model = AutoModelForCausalLM.from_pretrained('模型路径')

# 创建生成器
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer
)

# 输入文本并生成答案
total_answer = []
# for idx, text in enumerate(nlpcc_kbqa_test_knowques_compansw_new_inp):
for text in tqdm(nlpcc_kbqa_test_knowques_compansw_new_inp):
#     if idx <20:
    # text = '知识:别名是浙东行署社会教育工作队,中文名是浙东行署社会教育工作队,类型是越剧演出团体,骨干是竹方森、竺方渭、金桂芳,筹建时间是1943年7月,问题:浙东行署社会教育工作队何时建筑的?'
    generated = generator(text, max_length=512, num_return_sequences=1)[0]['generated_text']
    text_new = generated[len(text):].replace(' ', '')
    total_answer.append(text_new)
with open(nlpcc_kbqa_test_model_gene_answer_address, 'w', encoding='utf-8') as f:
    json.dump(total_answer, f, ensure_ascii=False)