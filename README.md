# SmokeNet

_A Lightweight Smoke Detector Network Based on Time-Series Sensors | 基于时间序列传感器的轻量级烟雾检测网络_

[中文说明 / Chinese version](README_zh.md)

## Overview
SmokeNet is a research-oriented PyTorch pipeline for detecting smoke events from multi-channel sensor sequences. It couples a compact 1D CNN with an LSTM backbone to capture local morphology and longer-term temporal dynamics, enabling deployment on resource-constrained platforms.

## Key Features
- 🪶 **Lightweight temporal modeling**：A 1D CNN + LSTM architecture that captures both local patterns and long-term dependencies.
- 📊 **Dual-task learning**：Supports fire event binary classification and optional fuel-type multi-class classification (`enable_fuel_classification`).
- 🧪 **Reproducible training**：Centralized configuration (`config/default.yaml`), fixed random seeds, and visualized metric outputs.
- 🧱 **Modular data loader**：A window-based `WindowDataset` for handling raw CSV time series and corresponding labels.
- 📈 **Evaluation utilities**：Built-in accuracy metrics, Top-K visualizations, and automatic saving of the best-performing model.

## Project Structure
- `main.py`: CLI entry point, responsible for configuration loading, data preparation, and switching between training and evaluation modes.
- `config/`: YAML configuration files (data, model, and training hyperparameters).
- `smokenet/`
  - `data/`: Windowed dataset implementation and CSV loading logic.
  - `models/`: Optional model builders and backbone implementations.
  - `utils/`: Utilities for logging, random seed control, plotting, and evaluation metrics.
  - `train.py`: Training loop, checkpoint saving, and history tracking.
- `dataset/`: Placeholder directory for default data and label files (CSV).

## Installation
1. Ensure Python 3.10+ is available.
2. Create environment (conda or venv). Example:
   ```bash
   conda env create -f environment.yml
   conda activate smokenet
   ```
   or use `pip install -r requirements.txt` to install dependencies.

## Data Preparation
- Place sensor sequences under `dataset/data/*.csv` and corresponding labels under `dataset/label/*.csv`.
- Each pair must share the same stem name (e.g., `sample001.csv`).
- Labels support per-timestep fire annotations (binary) and an optional fuel-class column that is constant across the sequence.
- Dataset splitting is controlled by `data.split_ratio` in the config.

## Quick Start
```bash
# Train
python main.py --mode train --config config/default.yaml

# Override key hyperparameters
python main.py --mode train --batch-size 16 --learning-rate 5e-4 --device cuda
```
- Training artifacts (weights, figures) are stored under `outputs/`.
- Fuel classification is enabled when both the dataset provides fuel labels and `model.enable_fuel_classification` is `true`.

## Configuration Highlights
Key fields in `config/default.yaml`:
- `data.window_size`: Sliding window length for sequences.
- `data.channels`: Expected sensor channels per timestep.
- `model.cnn_hidden`, `model.lstm_hidden`, `model.dropout`: Model capacity and regularization.
- `training.batch_size`, `training.learning_rate`, `training.num_epochs`: Optimization settings.

## Evaluation & Logging
- Metrics: fire accuracy (masked), optional fuel accuracy, loss curves.
- Checkpoints: best/last weights auto-saved to `outputs/weights/`.
- Visualizations: accuracy plots under `outputs/figures/`.

## Citation
If you build upon SmokeNet, please cite the repository and your configuration to ensure reproducibility.

---

Built with ❤️ by the SmokeNet contributors.