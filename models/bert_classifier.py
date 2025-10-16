import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_PATH)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.linear = nn.Linear(768, config.NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer