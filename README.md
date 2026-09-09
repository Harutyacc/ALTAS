# ALTAS

ALTAS 是一个基于实例级特征掩码、共享特征提取器和隐空间对抗约束的
合成数据特征选择实验。当前实现保留原有四组件模型和三阶段训练流程，
不再包含 t-SNE 分析。

## 项目结构

- `train.py`：实验入口及结果导出。
- `config.py`：数据、模型与训练配置。
- `data.py`：Syn1–Syn6 数据生成和 DataLoader 构造。
- `models.py`：掩码生成器、特征提取器、判别器和预测器。
- `masking.py`：批内打乱替换掩码。
- `trainer.py`：单批次的三阶段对抗训练。
- `training.py`：跨 epoch 训练与指标聚合。
- `evaluation.py`：测试集指标和特征报告。
- `visualization.py`：训练损失、保留率与准确率曲线。

## 运行

安装依赖，在 `config.py` 中调整配置，然后执行：

```powershell
python -m pip install -r requirements.txt
python train.py
```

每次运行的配置、日志、图表、指标、预测掩码和生成器权重会写入
`output/run_YYYYMMDD_HHMMSS_microseconds/`。
