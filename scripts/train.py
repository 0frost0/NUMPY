import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.data_loader import DataProcessor
from models.bert_classifier import BertClassifier
from training.trainer import Trainer
from utils.helpers import setup_seed


from torch.optim import Adam
from torch.nn import CrossEntropyLoss


def main():
    # 初始化配置
    config = Config()
    setup_seed(config.RANDOM_SEED)

    # 加载数据
    print("Loading data...")
    data_processor = DataProcessor(config)
    train_df, dev_df, test_df = data_processor.load_data()

    # 可视化数据分布（可选）
    # from utils.helpers import plot_text_length_distribution
    # plot_text_length_distribution(train_df)

    # 创建数据加载器
    train_loader, dev_loader, test_loader = data_processor.create_data_loaders(
        train_df, dev_df, test_df, config.BATCH_SIZE
    )

    # 初始化模型
    print("Initializing model...")
    model = BertClassifier(config)
    model.to(config.DEVICE)

    # 定义损失函数和优化器
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 训练模型
    print("Starting training...")
    trainer = Trainer(model, train_loader, dev_loader, criterion, optimizer, config.DEVICE, config)
    trainer.train()

    print("Training completed!")


if __name__ == "__main__":
    main()