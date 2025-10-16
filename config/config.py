
import torch


class Config:
    # 数据配置
    #DATASET_NAME = "sinhala-nlp/THUCNews"
    #BACKUP_DATASET = ("clue", "tnews")

    TRAIN_PATH = "/tmp/bert_train/data/data/train.txt"
    DEV_PATH = "/tmp/bert_train/data/data/dev.txt"   # 如果没有，可以自己从训练集中划分
    TEST_PATH = "/tmp/bert_train/data/data/test.txt"
    CLASS_PATH = "/tmp/bert_train/data/data/class.txt"

    # 模型配置
    BERT_PATH = '/tmp/bert_train/models/Bert_to_Chinese'
    MAX_LENGTH = 35
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 10

    # 训练配置
    EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-5
    RANDOM_SEED = 1999

    # 设备配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 路径配置
    SAVE_PATH = './outputs/checkpoints/'
    LOG_PATH = './outputs/logs/'