# SmokeNet

_基于时间序列传感器的轻量级烟雾检测网络 | A Lightweight Smoke Detector Network Based on Time-Series Sensors_

[English version / 英文版](README.md)

## 概览
SmokeNet 是一个以研究为导向的 PyTorch 流水线，用于从多通道传感器序列中识别烟雾事件。模型采用紧凑的 1D CNN 与 LSTM 组合，既捕获局部形态又建模长时程动态，适合在资源受限设备上部署。

## 核心特性
- 🪶 **轻量化时序建模**：1D CNN + LSTM 架构兼顾局部模式和长期依赖。
- 📊 **双任务学习**：支持火情二分类与可选燃料类型多分类（`enable_fuel_classification`）。
- 🧪 **可复现训练**：集中配置文件、固定随机种子、可视化指标输出。
- 🧱 **模块化数据加载**：基于窗口的 `WindowDataset` 读取原始 CSV 序列与标签对。
- 📈 **评估与可视化**：内置精度、Top-K 可视化与最佳模型自动保存。

## 项目结构
- `main.py`: 命令行入口，完成配置加载、数据准备与训练/评估切换。
- `config/`: YAML 配置（数据、模型、训练超参）。
- `smokenet/`
  - `data/`: CSV 读取与滑窗数据集实现。
  - `models/`: 模型构建器与主干实现。
  - `utils/`: 日志、随机种子、绘图与评估指标工具。
  - `train.py`: 训练循环、检查点保存与历史记录。
- `dataset/`: 默认数据与标签目录占位（CSV）。

## 环境安装
1. 确保已安装 Python 3.10+。
2. 创建虚拟环境（conda 或 venv）。示例：
   ```bash
   conda env create -f environment.yml
   conda activate smokenet
   ```
   或使用 `pip install -r requirements.txt` 安装依赖。

## 数据准备
- 将传感器序列放置于 `dataset/data/*.csv`，对应标签放置于 `dataset/label/*.csv`。
- 数据与标签文件需同名（如 `sample001.csv`）。
- 标签支持逐时间步火情（二分类）以及可选的恒定燃料类别列。
- 数据集划分由配置项 `data.split_ratio` 控制。

## 快速开始
```bash
# 训练 | Train
python main.py --mode train --config config/default.yaml

# 覆盖主要超参 | Override key hyperparameters
python main.py --mode train --batch-size 16 --learning-rate 5e-4 --device cuda
```
- 训练产物（权重、图像）保存在 `outputs/` 目录。
- 当数据集提供燃料标签且开启 `model.enable_fuel_classification` 时，将启用燃料分类。

## 配置要点
- `data.window_size`: 序列滑窗长度。
- `data.channels`: 每时间步的传感器通道数。
- `model.cnn_hidden`, `model.lstm_hidden`, `model.dropout`: 模型容量与正则化。
- `training.batch_size`, `training.learning_rate`, `training.num_epochs`: 优化器设置与训练轮数。

## 评估与日志
- 指标：火情精度（掩码）、可选的燃料精度与损失曲线。
- 检查点：最佳与最新权重自动保存在 `outputs/weights/`。
- 可视化：精度曲线保存在 `outputs/figures/`。

## 引用
若在研究中使用 SmokeNet，请引用本仓库并明确使用的配置以保证结果可复现。

---

由 SmokeNet 贡献者倾力打造。