import torch
from torch.optim import Adam
from tqdm import tqdm
import os

from utils.helpers import save_model


class Trainer:
    def __init__(self, model, train_loader, dev_loader, criterion, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.best_dev_acc = 0

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_acc_train = 0
        total_loss_train = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            acc = (outputs.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()

        return total_loss_train, total_acc_train

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_acc = 0
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)

                acc = (outputs.argmax(dim=1) == labels).sum().item()
                total_acc += acc
                total_loss += loss.item()

        return total_loss, total_acc

    def train(self):
        """完整训练过程"""
        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}")

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            dev_loss, dev_acc = self.evaluate(self.dev_loader)

            # 打印结果
            train_samples = len(self.train_loader.dataset)
            dev_samples = len(self.dev_loader.dataset)

            print(f"Train Loss: {train_loss / train_samples:.3f} | "
                  f"Train Accuracy: {train_acc / train_samples:.3f}")
            print(f"Dev Loss: {dev_loss / dev_samples:.3f} | "
                  f"Dev Accuracy: {dev_acc / dev_samples:.3f}")

            # 保存最佳模型
            current_dev_acc = dev_acc / dev_samples
            if current_dev_acc > self.best_dev_acc:
                self.best_dev_acc = current_dev_acc
                save_model(self.model, self.config.SAVE_PATH, 'best.pt')
                print(f"New best model saved with accuracy: {self.best_dev_acc:.3f}")

        # 保存最终模型
        save_model(self.model, self.config.SAVE_PATH, 'last.pt')
        print("Training completed!")