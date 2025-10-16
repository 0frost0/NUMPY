import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model, save_path, save_name):
    """保存模型"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


def plot_text_length_distribution(df, save_path=None):
    """绘制文本长度分布图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    length_counts = df['text'].apply(len).value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.hist(length_counts.index, bins=len(length_counts), weights=length_counts.values)
    plt.xlabel('文本长度')
    plt.ylabel('频数')
    plt.title('字符串长度分布直方图')

    if save_path:
        plt.savefig(save_path)
    plt.show()