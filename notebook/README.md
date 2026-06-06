# Reproduction Notebook

`REPRODUCE_RESULTS.ipynb` 用于一键复现当前 Chronos / TimesFM 实验结果。

`EVALUATE_RESULTS.ipynb` 用于直接加载现有基模或 `output/` 中已有 checkpoint 做 eval 和预测展示，不运行 pretrain 或 fine-tune。

## 用法

1. 打开 `REPRODUCE_RESULTS.ipynb`。
2. 先运行第一个 setup cell。
3. 按需要运行后续实验 cell。每个实验 cell 都会：
   - 在 `output/notebook_repro_时间戳/` 下生成独立输出目录；
   - 打印 `results_summary.json`；
   - 随机抽取一条序列展示预测图。

如果只想重新评估已有 checkpoint，打开 `EVALUATE_RESULTS.ipynb`，先运行 setup cell，再运行 Chronos / TimesFM eval cell。它会写入 `output/notebook_eval_时间戳/`。

## 基模权重

基座模型权重不保存在本仓库。

- Chronos 使用 `amazon/chronos-t5-base`。
- TimesFM 使用 `google/timesfm-2.5-200m-pytorch`。
- `timesfm/` 目录只是 TimesFM 源码，不包含 `model.safetensors` 权重。
- 权重会从 Hugging Face cache 读取；如果本地没有缓存，脚本会尝试下载。
- 常见缓存位置是 `$HUGGINGFACE_HUB_CACHE`、`$HF_HOME/hub` 或 `~/.cache/huggingface/hub`。
- 如果在离线环境复现，需要先把上述两个模型放进 Hugging Face cache，或者设置 `HF_HOME` / `HUGGINGFACE_HUB_CACHE` 指向已有缓存。

## 数据文件

数据来自 [Monash Time Series Forecasting Archive](https://forecastingdata.org/)。实验默认读取 `data/extracted/` 下的 Monash `.tsf` 文件，例如：

- `data/extracted/tourism_monthly_dataset.tsf`
- `data/extracted/m4_monthly_dataset.tsf`
- `data/extracted/m4_daily_dataset.tsf`
- `data/extracted/m4_hourly_dataset.tsf`
- `data/extracted/m3_monthly_dataset.tsf`
- `data/extracted/m1_monthly_dataset.tsf`
- `data/extracted/weather_dataset.tsf`
- `data/extracted/electricity_hourly_dataset.tsf`
- `data/extracted/traffic_hourly_dataset.tsf`
- `data/extracted/temperature_rain_dataset_without_missing_values.tsf`
- `data/extracted/kaggle_web_traffic_dataset_without_missing_values.tsf`
- `data/extracted/rideshare_dataset_without_missing_values.tsf`
- `data/extracted/vehicle_trips_dataset_without_missing_values.tsf`
- `data/extracted/kdd_cup_2018_dataset_without_missing_values.tsf`
- `data/extracted/covid_deaths_dataset.tsf`
- `data/extracted/nn5_daily_dataset_without_missing_values.tsf`
- `data/extracted/fred_md_dataset.tsf`

如果缺少数据，普通 Chronos / TimesFM 脚本通常会在读取 `.tsf` 时直接报 `FileNotFoundError`；TimesFM staged pretrain 会报更明确的 `Dataset not found: ...`。

解决方式：

- 确认从项目根目录运行 notebook，或先运行第一个 setup cell 让 notebook 自动定位项目根。
- 确认上述 `.tsf` 文件存在于 `data/extracted/`。
- 如果数据被删掉，需要从 [Monash Time Series Forecasting Archive](https://forecastingdata.org/) 重新下载并解压到 `data/extracted/`，保持文件名不变。
- Chronos 会复用 `data/arrow_training/<dataset>/` 下已生成的 Arrow 文件；如果怀疑 Arrow 与 `.tsf` 不一致，可以删除对应数据集的 Arrow 子目录后重跑。

## TimesFM 复现运行顺序

在 `REPRODUCE_RESULTS.ipynb` 中，TimesFM 的 direct / LoRA 实验依赖当前输出目录里的 pretrain checkpoint；不会回退旧 output。

- `timesfm_original_head_eval` / `timesfm_original_head_lora` 之前先跑 `timesfm_original_head_pretrain`
- `timesfm_ce64_eval` / `timesfm_ce64_lora_*` 之前先跑 `timesfm_ce64_pretrain`
- `timesfm_ce256_eval` / `timesfm_ce256_lora_*` 之前先跑 `timesfm_ce256_pretrain`
- `timesfm_new64_eval` / `timesfm_new64_lora` 之前先跑 `timesfm_new64_pretrain`

`EVALUATE_RESULTS.ipynb` 不需要上述训练顺序；它直接读取 `output/` 中固定路径下已有的 checkpoint。

## 参考结果

Chronos 结果大致如下：

| Model | WQL | MASE |
|---|---:|---:|
| Zero-shot | 1.5441 | 1.6617 |
| CE+LoRA | 1.2244 | 1.3806 |
| bin-MSE+LoRA | 1.2651 | 1.4176 |
| bin-MSE+CE+LoRA, lambda=1e-4 | 1.2638 | 1.4147 |
| bin-W1+LoRA | 1.2750 | 1.3151 |
| bin-W2+LoRA | 1.3252 | 1.3352 |
| bin-CRPS+LoRA | 1.2325 | 1.3713 |
| bin-OrdinalCE+LoRA | 1.2357 | 1.3765 |
| bin-Huber(16)+LoRA | 1.2562 | 1.3803 |

TimesFM 结果大致如下：

| Model | WQL | MASE |
|---|---:|---:|
| Base zero-shot | 1.1899 | 1.1688 |
| Base LoRA | 1.2058 | 1.1410 |
| Original head direct | 1.2795 | 1.2389 |
| Original head LoRA | 1.2049 | 1.1468 |
| CE64 direct | 1.4605 | 1.3809 |
| CE64 LoRA low LR | 1.2308 | 1.1162 |
| CE256 direct | 1.3624 | 1.2862 |
| CE256 LoRA head_lr=1e-5 | 1.2691 | 1.2336 |
| CE256 LoRA head_lr=5e-5 | 1.3094 | 1.2605 |
| new64 large direct | 4.5760 | 5.5845 |
| new64 large LoRA | 1.4641 | 1.3968 |
