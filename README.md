# CACAM

**CACAM (Channel-Aware Multivariate Time Series Anomaly Detection)**

This repository provides the CACAM model code and benchmark scripts for multivariate time-series anomaly detection.

## 原仓库引用

本项目是在以下仓库基础上整理的：
- Original repository: https://github.com/decisionintelligence/CATCH
- 在原始仓库基础上进行了重命名与重构说明整理，以便与你的 `CACAM` 工程对齐。

## 目录说明

- `ts_benchmark/`：基准测试主流程与模型入口
- `scripts/`：各数据集/任务脚本
- `run_cacam.sh`：快速运行入口（按训练/评估场景执行）
- `run_cacam_hpo.sh`：超参数搜索/实验脚本
- `requirements.txt`：依赖清单
- `docs/`：说明与图示
- `dataset/`：数据目录（未提交）
- `result/`：实验输出目录（未提交）

## 快速开始

```bash
pip install -r requirements.txt
```

```bash
sh run_cacam.sh
```

```bash
sh run_cacam_hpo.sh
```

## 结果目录

实验输出默认落在 `result/`；如需复现实验环境，请确保 `dataset/` 与 `result/` 已按运行脚本路径准备。

## Contributors

- sylviaqs (owner)
- ZengYuXiang7

## Contribution

如果这个仓库对你有帮助，请保留原始仓库出处并在发布时注明来源。
