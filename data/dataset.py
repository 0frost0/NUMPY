import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config.config import Config


class TextClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=35):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        label = self.df.iloc[idx]['label']

        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.df)