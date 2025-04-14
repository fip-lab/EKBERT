import json
import torch
from transformers import BertTokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np


# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


tokenizer = BertTokenizer.from_pretrained('../gpt2_model')
model = GPT2LMHeadModel.from_pretrained('../gpt2_model')


nlpcc2018_kbqa_train_knowques_compansw_new_address = '../data/nlpcc2018_kbqa_train_knowques_compansw_single.json'
nlpcc2018_kbqa_dev_knowques_compansw_new_address = '../data/nlpcc2018_kbqa_dev_knowques_compansw_single.json'
nlpcc2018_kbqa_test_knowques_compansw_new_address = '../data/nlpcc2018_kbqa_test_knowques_compansw_single.json'


with open(nlpcc2018_kbqa_train_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    nlpcc2018_train_data = json.load(f)
with open(nlpcc2018_kbqa_dev_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    nlpcc2018_dev_data = json.load(f)


train_data = nlpcc2018_train_data
dev_data = nlpcc2018_dev_data

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
train_dataset = TextDataset(train_data, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

dev_dataset = TextDataset(dev_data, tokenizer, max_length)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader):
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

        train_loss += loss.item()

    # 输出损失
    train_avg_loss = train_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs} - train_loss: {train_avg_loss:.4f}')

    model.eval()
    dev_loss = 0
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            # 将数据移动到 GPU 上
            input_ids = batch[0].to(device)
            attn_masks = batch[1].to(device)

            # 计算模型输出
            outputs = model(input_ids, attention_mask=attn_masks, labels=input_ids)
            loss = outputs.loss

            # 统计损失和准确度
            dev_loss += loss.item()

    # 计算平均损失和准确度
    dev_avg_loss = dev_loss / len(dev_dataloader)

    # 输出损失
    print(f'Epoch {epoch + 1}/{num_epochs} - dev_loss: {dev_avg_loss:.4f}')

# 保存模型
output_dir = '../gpt2_trained_model/gpt2_trained_model_nlpcc2018'
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)


