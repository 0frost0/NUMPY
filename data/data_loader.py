import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from typing import List, Tuple

# ------------------- Dataset -------------------
class TextClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_length: int, label_map: dict):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "text"]
        label_str = str(self.df.loc[idx, "label"]).strip()
        if label_str.isdigit():
            label_id = int(label_str)  # 直接使用train.txt中的数字标签
        else:
            label_id = self.label_map[label_str]  # 如果是汉字标签则用映射表

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long)
        }

# ------------------- DataProcessor -------------------
class DataProcessor:
    """
    功能：
    1. 读取 txt 文件 → DataFrame
    2. 构建 DataLoader
    3. 使用 class.txt 映射标签字符串到整数
    """

    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)
        self.label_map = self._load_labels(self.config.CLASS_PATH)

    # --------- public ---------
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = self._read_single_file(self.config.TRAIN_PATH, label_first=False)
        dev_df   = self._read_single_file(self.config.DEV_PATH, label_first=False)
        test_df  = self._read_single_file(self.config.TEST_PATH, label_first=False)
        return train_df, dev_df, test_df

    def create_data_loaders(
        self,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        test_df: pd.DataFrame,
        batch_size: int = 64,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = self._build_loader(train_df, batch_size, shuffle=True)
        dev_loader   = self._build_loader(dev_df,   batch_size, shuffle=False)
        test_loader  = self._build_loader(test_df,  batch_size, shuffle=False)
        return train_loader, dev_loader, test_loader

    # --------- private ---------
    @staticmethod
    def _read_single_file(
        path: str,
        sep: str = "\t",
        label_first: bool = True,
        dropna: bool = True,
    ) -> pd.DataFrame:
        records: List[Tuple[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(sep, 1)
                if len(parts) != 2:
                    continue
                left, right = parts
                text, label = (right, left) if label_first else (left, right)
                records.append((text, label))

        df = pd.DataFrame(records, columns=["text", "label"])
        if dropna:
            df.dropna(subset=["text", "label"], inplace=True)
        return df.reset_index(drop=True)

    def _build_loader(self, df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TextClassificationDataset(df, self.tokenizer, self.config.MAX_LENGTH, self.label_map)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def _load_labels(path: str) -> dict:
        """
        从 class.txt 生成 label_map
        输出: {"finance":0, "realty":1, ...}
        """
        with open(path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        return {label: idx for idx, label in enumerate(labels)}
