import torch
import copy
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BertTokenizer, AdamW
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
# from torch.optim import Adam

class CustomDataset(Dataset):
    def __init__(self, texts, max_input_length, max_output_length, tokenizer):
        self.texts = texts
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_text = text['input']
        target_text = text['target']

        # Tokenize and encode the input and target texts
        input_encoding = self.tokenizer(input_text, max_length=self.max_input_length, truncation=True, padding='max_length', return_tensors='pt')
        target_encoding = self.tokenizer(target_text, max_length=self.max_output_length, truncation=True, padding='max_length', return_tensors='pt')

        labels = copy.deepcopy(target_encoding['input_ids'].squeeze())
        labels[labels == 0] = -100

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'decoder_input_ids': target_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze(),
            'labels': labels
        }

class BARTFineTuningModel(LightningModule):
    def __init__(self, model_name, tokenizer_name, train_dataset, val_dataset, batch_size=2):
        super(BARTFineTuningModel, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        decoder_input_ids = batch['decoder_input_ids']
        decoder_attention_mask = batch['decoder_attention_mask']
        labels = batch['labels']

        outputs = self(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        decoder_input_ids = batch['decoder_input_ids']
        decoder_attention_mask = batch['decoder_attention_mask']
        labels = batch['labels']

        outputs = self(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels)
        loss = outputs.loss
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=3e-5, eps=1e-8)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Example usage
# train_texts = [{'input': 'input text 1', 'target': 'target text 1'}, {'input': 'input text 2', 'target': 'target text 2'}]
# val_texts = [{'input': 'input text 3', 'target': 'target text 3'}, {'input': 'input text 4', 'target': 'target text 4'}]

nlpcc_kbqa_train_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_train_knowques_compansw.json'
nlpcc_kbqa_dev_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_dev_knowques_compansw.json'
nlpcc_kbqa_test_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_test_knowques_compansw.json'


with open(nlpcc_kbqa_train_knowques_compansw_address, 'r', encoding='utf-8') as f:
    nlpcc_train_data = json.load(f)
with open(nlpcc_kbqa_dev_knowques_compansw_address, 'r', encoding='utf-8') as f:
    nlpcc_dev_data = json.load(f)

train_dataset = CustomDataset(texts=nlpcc_train_data, max_input_length=512, max_output_length=512, tokenizer=BertTokenizer.from_pretrained('../model_files_chinese'))
val_dataset = CustomDataset(texts=nlpcc_dev_data, max_input_length=512, max_output_length=512, tokenizer=BertTokenizer.from_pretrained('../model_files_chinese'))


if __name__ == '__main__':
    model = BARTFineTuningModel(
        model_name='../model_files_chinese',
        tokenizer_name='../model_files_chinese',
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=6
    )
    trainer = Trainer(max_epochs=1, gpus=[1])  # You can customize the Trainer as needed
    trainer.fit(model)
    torch.save(model, '../bart_trained_model/bart_nlpcc1.pkl')