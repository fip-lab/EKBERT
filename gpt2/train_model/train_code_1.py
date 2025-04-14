# from transformers import AutoTokenizer
# from transformers import Trainer, TrainingArguments
# from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
# from transformers import pipeline
import json
import torch
from transformers import BertTokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('../gpt2_model')
model = GPT2LMHeadModel.from_pretrained('../gpt2_model')


nlpcc_kbqa_train_knowques_compansw_new_address = '../data/nlpcc_kbqa_train_knowques_compansw_single.json'
nlpcc_kbqa_dev_knowques_compansw_new_address = '../data/nlpcc_kbqa_dev_knowques_compansw_single.json'
nlpcc_kbqa_test_knowques_compansw_new_address = '../data/nlpcc_kbqa_test_knowques_compansw_single.json'

ccks2020_kbqa_train_knowques_compansw_new_address = '../data/ccks2020_kbqa_train_knowques_compansw_single.json'
ccks2020_kbqa_dev_knowques_compansw_new_address = '../data/ccks2020_kbqa_dev_knowques_compansw_single.json'
ccks2020_kbqa_test_knowques_compansw_new_address = '../data/ccks2020_kbqa_test_knowques_compansw_single.json'

ccks2022_kbqa_train_knowques_compansw_new_address = '../data/ccks2022_kbqa_train_knowques_compansw_single.json'
ccks2022_kbqa_dev_knowques_compansw_new_address = '../data/ccks2022_kbqa_dev_knowques_compansw_single.json'
ccks2022_kbqa_test_knowques_compansw_new_address = '../data/ccks2022_kbqa_test_knowques_compansw_single.json'


with open(nlpcc_kbqa_train_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    nlpcc_train_data = json.load(f)
with open(nlpcc_kbqa_dev_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    nlpcc_dev_data = json.load(f)
with open(nlpcc_kbqa_test_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    nlpcc_test_data = json.load(f)
with open(ccks2020_kbqa_train_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2020_train_data = json.load(f)
with open(ccks2020_kbqa_dev_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2020_dev_data = json.load(f)
with open(ccks2020_kbqa_test_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2020_test_data = json.load(f)
with open(ccks2022_kbqa_train_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2022_train_data = json.load(f)
with open(ccks2022_kbqa_dev_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2022_dev_data = json.load(f)
with open(ccks2022_kbqa_test_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2022_test_data = json.load(f)

train_data = nlpcc_train_data + ccks2020_train_data + ccks2022_train_data
dev_data = nlpcc_dev_data + ccks2020_dev_data + ccks2022_dev_data

# 定义数据集类
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        for text in data:
            # 编码文本
            encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.input_ids.append(encoded_dict['input_ids'])
            self.attn_masks.append(encoded_dict['attention_mask'])

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attn_masks = torch.cat(self.attn_masks, dim=0)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


# 加载模型和分词器
# model_name = 'uer/gpt2-chinese-small'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# 将模型移到 GPU 上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义训练参数
batch_size = 1
max_length = 512
learning_rate = 3e-4
num_epochs = 5

# 加载训练数据
# train_data = ['训练数据1', '训练数据2', '训练数据3']
dataset = TextDataset(train_data, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(dataloader):
        # 将数据移动到 GPU 上
        input_ids = batch[0].to(device)
        attn_masks = batch[1].to(device)

        # 计算模型输出
        outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)
        loss = outputs.loss

        # 反向传播和更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 输出损失
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs} - loss: {avg_loss:.4f}')

# 保存模型
output_dir = '../gpt2_trained_model'
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)


