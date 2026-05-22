# CACAM Causal Discovery Benchmark

本仓库是一个面向**多变量时间序列异常检测**的统一实验框架，核心目标是：
- 在同一套数据加载、训练/推理、评估、报表流程下，对比不同检测模型；
- 支持 `detect_score`（异常分数）与 `detect_label`（异常标签）两类任务；
- 支持自研模型（如 CACAM/CATCH 等）与第三方模型（Merlion/TODS/Time-Series-Library）并行对比。

仓库当前重点是 anomaly detection，forecasting 相关代码也保留在框架中。

---

## 1. 项目结构（按职责）

```text
cacam-causal-discovery/
├── config/                          # 任务配置（策略、指标、数据集过滤、report）
├── dataset/
│   └── anomaly_detect/              # 异常检测数据与元信息（DETECT_META.csv）
├── scripts/
│   ├── run_benchmark.py             # 主入口（CLI）
│   ├── CACAM.sh / CACAM-FFT.sh      # 顶层快捷脚本
│   └── multivariate_detection/
│       ├── detect_score/.../*.sh    # 分数任务脚本
│       └── detect_label/.../*.sh    # 标签任务脚本
├── ts_benchmark/
│   ├── pipeline.py                  # 数据->模型->评估->日志 的主流水线
│   ├── data/                        # 数据源、数据池、元数据管理
│   ├── models/                      # 模型加载与超参数装配
│   ├── evaluation/                  # strategy + metrics + evaluator
│   ├── report/                      # leaderboard 汇总输出
│   └── baselines/                   # 各模型实现与 adapter
├── result/                          # 实验输出目录
├── run_cacam.sh                     # 批量跑 CACAM
├── run_cacam_fft_mix.sh             # 批量跑 CACAM_FFT_mix
└── requirements.txt
```

---

## 2. 端到端执行链路

### 2.1 CLI 入口
运行命令统一进入：

```bash
python ./scripts/run_benchmark.py --config-path ... --model-name ... --model-hyper-params ...
```

`run_benchmark.py` 的主要职责：
1. 读取 `config/*.json`。
2. 将 CLI 参数覆盖到 `data_config / model_config / evaluation_config / report_config`。
3. 初始化并行后端（`sequential` 或 `ray`）。
4. 调用 `ts_benchmark.pipeline.pipeline(...)` 执行实验。
5. 将日志写入 `result/`，再生成 leaderboard CSV。

### 2.2 Pipeline 主流程
`ts_benchmark/pipeline.py` 中的 `pipeline()` 分四步：
1. **Data**：根据 `data_set_name` 选择数据源（当前检测任务常用 `large_detect` -> `LocalAnomalyDetectDataSource`）。
2. **Model**：调用 `get_models(model_config)` 解析模型名、adapter、超参数，构建 `ModelFactory`。
3. **Evaluation**：按策略（如 `unfixed_detect_score`）调度评测任务。
4. **Recording**：把每个模型的结果 DataFrame 持久化到 `result/<save_path>/`。

### 2.3 评估调度
`evaluation/evaluate_model.py`：
- 根据 `strategy_name` 从 `STRATEGY` 注册表取策略类。
- 根据策略允许的指标做合法性校验。
- 每个 `(model, series)` 组合调度一次 `strategy.execute(...)`。
- 结果统一补齐字段（模型名、策略参数、模型参数等）后写入日志。

---

## 3. 数据层架构

### 3.1 数据目录约定
检测任务使用：
- `dataset/anomaly_detect/data/*.csv`
- `dataset/anomaly_detect/DETECT_META.csv`

`DETECT_META.csv` 中包含重要字段（示例）：
- `file_name`
- `size`（`large/small/user`）
- `train_lens`（unfixed 切分时用于 train/test 分割）
- 其他统计或标签字段

### 3.2 DataSource 与 DataPool
- `LocalAnomalyDetectDataSource`：负责并行加载数据文件和可选协变量。
- `GlobalStorageDataServer`：把加载后的数据挂到全局共享存储。
- `DataPool`（单例）：策略执行时通过 `DataPool().get_pool().get_series(...)` 获取数据。

这套设计让策略代码不依赖具体文件路径，只依赖统一数据接口。

---

## 4. 模型层架构

### 4.1 模型发现与导入
模型名通过以下顺序解析（`models/model_loader.py`）：
1. `global.xxx`（如果使用 global 前缀）
2. `ts_benchmark.baselines.<model_name>`
3. `<model_name>` 直接 import

例如：
- `self_impl.CACAM`
- `catch.CATCH`
- `time_series_library.TimesNet`
- `merlion.AutoEncoder`
- `tods.ocsvmski`

### 4.2 ModelFactory 机制
每个模型最终被包装成 `ModelFactory`，评估时按需实例化，避免状态污染。

### 4.3 超参数装配规则
超参数由两部分合成：
1. `recommend_model_hyper_params`（来自 config，全局推荐值）
2. `model_hyper_params`（命令行/脚本显式传入，优先级更高）

如果模型声明了 `required_hyper_params`，且未被前两者补齐，会直接报错。

### 4.4 Adapter 机制（关键）
`ADAPTER` 注册在 `ts_benchmark/baselines/__init__.py`。
当前检测任务里最常见的是：
- `transformer_adapter`：把 `time_series_library.*` 模型适配为检测接口。

它会为 Transformer 家族注入必要字段（如 `seq_len/horizon/norm`）。

---

## 5. 策略层（score / label / both）

策略注册在 `evaluation/strategy/__init__.py`，核心检测策略有：
- `unfixed_detect_score`
- `unfixed_detect_label`
- `unfixed_detect_both`
- `fixed_detect_*`
- `all_detect_*`

### 5.1 Unfixed 与 Fixed 的区别
- `unfixed_detect_*`：按 `DETECT_META.csv` 的 `train_lens` 切分 train/test。
- `fixed_detect_*`：按配置里的 `train_test_split` 比例切分。
- `all_detect_*`：训练和测试都使用全量序列（多用于某些对比场景）。

### 5.2 Score 与 Label 的接口约定
模型优先实现：
- `detect_fit(train_data, train_label_or_test_label)`
- `detect_score(test_data)`
- `detect_label(test_data)`

若不存在 `detect_fit`，框架会回退调用 `fit(...)`。

---

## 6. 指标与结果产物

### 6.1 指标来源
- score 类指标：`evaluation/metrics/classification_metrics_score.py`
  - 如 `auc_roc`, `auc_pr`, `VUS_ROC`, `VUS_PR`
- label 类指标：`evaluation/metrics/classification_metrics_label.py`
  - 如 `f_score`, `affiliation_f` 等

### 6.2 日志字段
记录字段定义在 `evaluation/strategy/constants.py`，包括：
- `model_name`
- `model_params`
- `strategy_args`
- `file_name`
- `fit_time`, `inference_time`
- `typical_anomaly_ratio`
- `actual_data`, `inference_data`, `log_info`

### 6.3 报表生成
`report/report_csv.py` + `report/utils/leaderboard.py`：
- 从多个日志 CSV 聚合；
- 按指标透视和聚合（默认 `mean`）；
- 对缺失值按策略填充；
- 输出 leaderboard 到 `result/<save_path>/xxx_report.csv`。

---

## 7. 你当前脚本体系（multivariate_detection）

`scripts/multivariate_detection/` 下按任务拆分：
- `detect_score/<dataset>_script/*.sh`
- `detect_label/<dataset>_script/*.sh`

每个脚本通常固定：
- `--config-path`（score/label 对应不同 config）
- `--data-name-list`（单数据集）
- `--model-name`
- `--model-hyper-params`
- `--save-path`

这使得你可以按“数据集 × 模型 × 任务类型”批量管理实验。

---

## 8. 快速开始

### 8.1 环境
```bash
pip install -r requirements.txt
```

### 8.2 运行单模型单数据集
```bash
python ./scripts/run_benchmark.py \
  --config-path "unfixed_detect_score_multi_config.json" \
  --data-name-list "SMD.csv" \
  --model-name "catch.CATCH" \
  --model-hyper-params '{"num_epochs": 3}' \
  --gpus 0 \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "score/CATCH"
```

### 8.3 运行批量脚本
```bash
bash run_cacam.sh
bash run_cacam_fft_mix.sh
```

---

## 9. 常见扩展点

### 9.1 新增模型
1. 在 `ts_benchmark/baselines/...` 添加实现。
2. 确保可通过 `model_name` 被 import。
3. 实现检测接口（`detect_fit/detect_score/detect_label`）或兼容 `fit/predict`。
4. 如需框架补参，声明 `required_hyper_params`。

### 9.2 新增策略
1. 在 `evaluation/strategy/` 新建策略类，继承 `Strategy`。
2. 实现 `execute()`、`accepted_metrics()`、`field_names`。
3. 在 `evaluation/strategy/__init__.py` 注册名称。

### 9.3 新增指标
1. 在 `evaluation/metrics/` 增加函数。
2. 加入对应模块的 `__all__`。
3. 策略的 `accepted_metrics()` 自动可见。

---

## 10. 设计特点与取舍

### 优点
- 统一实验协议：数据、模型、策略、报表解耦。
- 脚本化程度高：易于大规模网格实验与复现实验。
- 对多模型生态友好：自研 + 第三方库可并行纳入。

### 当前取舍
- `scripts/multivariate_detection` 下脚本数量大，参数管理成本高。
- 部分模型 heavily tuned，跨任务参数不一致（这是你已观察到的问题）。
- 结果里包含 pickle/base64 字段，便于复现但不适合直接人工阅读。

---

## 11. 术语约定

- `detect_score`：输出连续异常分数，再按阈值/指标评估。
- `detect_label`：直接输出二值异常标签。
- `unfixed`：训练长度由元数据 `train_lens` 决定。
- `both`：同次评估里同时计算 score 与 label 指标。
