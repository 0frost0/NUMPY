"""
BERT文本分类项目
"""

__version__ = "1.0.0"
__author__ = "SHITAO DENG"

from config import Config
from data import DataProcessor, TextClassificationDataset
from models import BertClassifier
from training import Trainer
from utils import setup_seed, save_model, plot_text_length_distribution

__all__ = [
    'Config',
    'DataProcessor',
    'TextClassificationDataset',
    'BertClassifier',
    'Trainer',
    'setup_seed',
    'save_model',
    'plot_text_length_distribution'
]