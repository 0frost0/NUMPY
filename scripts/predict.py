import sys
import os
import torch
from transformers import BertTokenizer
from config.config import Config
from models.bert_classifier import BertClassifier
from utils import setup_seed


class Predictor:
    def __init__(self, model_path, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)
        self.model = self.load_model(model_path)
        self.label_list = self._load_labels(config.CLASS_PATH)  # ← 新增：加载class.txt

    def _load_labels(self, class_path):
        """读取 class.txt 中的标签（每行一个类别名）"""
        with open(class_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        return labels

    def load_model(self, model_path):
        model = BertClassifier(self.config)
        model.load_state_dict(torch.load(model_path, map_location=self.config.DEVICE))
        model.to(self.config.DEVICE)
        model.eval()
        return model

    def predict(self, text):
        """输入一条文本，返回类别索引与类别名称"""
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].to(self.config.DEVICE)
        attention_mask = encoding['attention_mask'].to(self.config.DEVICE)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            prediction = torch.argmax(outputs, dim=1).item()

        label_name = self.label_list[prediction] if prediction < len(self.label_list) else "未知类别"
        return prediction, label_name


def main():
    setup_seed(Config.RANDOM_SEED)
    config = Config()

    # 模型路径
    model_path = os.path.join(config.SAVE_PATH, "best.pt")
    predictor = Predictor(model_path, config)

    print("模型加载完成，可以开始输入文本进行预测（输入空行退出）\n")

    while True:
        text = input("新闻标题：")
        if not text.strip():
            break
        pred_id, pred_label = predictor.predict(text)
        print(f"预测类别编号: {pred_id}")
        print(f"预测类别名称: {pred_label}\n")


if __name__ == "__main__":
    main()
