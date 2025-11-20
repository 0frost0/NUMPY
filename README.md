# BERT文本分类项目

基于BERT的中文文本分类项目，使用THUCNews数据集。

## 项目结构
bert-classification/
├── config/ # 配置文件
├── data/ # 数据加载和处理
├── models/ # 模型定义
├── training/ # 训练逻辑
├── utils/ # 工具函数
├── scripts/ # 运行脚本
└── outputs/ # 输出文件

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
## 使用说明

1. **创建项目目录**：按照上面的结构创建文件夹和文件
2. **安装依赖**：`pip install -r requirements.txt`
3. **运行训练**：直接运行 `python scripts/train.py`
4. **模块化导入**：每个模块都可以独立导入和使用

这样的结构使得代码更加：
- **模块化**：功能分离，易于维护
- **可配置**：所有超参数集中在config中
- **可扩展**：易于添加新功能
- **可重用**：组件可以在其他项目中重用

