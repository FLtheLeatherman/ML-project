# 当前保留数据集

本项目数据来自 [Monash Time Series Forecasting Archive](https://forecastingdata.org/)。实际数据文件不提交到 Git；clone 后需要自行从该网站下载并解压。当前本地只保留复现实验需要的 Monash `.tsf` 数据。原始 zip 与解压后的 `.tsf` 文件一一对应：

- zip: `data/<dataset>.zip`
- tsf: `data/extracted/<dataset>.tsf`

## 保留原则

保留清单来自当前文档与 `notebook/REPRODUCE_RESULTS.ipynb` 的实验矩阵：

- Chronos / TimesFM 下游评估使用 `tourism_monthly_dataset`。
- TimesFM original head、CE64、CE256 的预训练池使用 M 系列、weather、电力和交通数据。
- TimesFM `new64 large` 复现会调用 staged pretrain，并自动经过 starter bootstrap，因此保留 starter / large 两个 stage 的数据池。
- `monthly_align` 不是当前 notebook 主线复现实验，已从 staged pretrain 当前预设中移除。

## 保留列表

| 数据集 | 用途 |
|---|---|
| `tourism_monthly_dataset` | Chronos / TimesFM 下游评估与 LoRA 微调 |
| `m4_monthly_dataset` | TimesFM original head / CE64 / CE256 / staged pretrain |
| `m4_daily_dataset` | TimesFM original head / CE64 / CE256 / staged pretrain |
| `m4_hourly_dataset` | TimesFM staged large pretrain |
| `m3_monthly_dataset` | TimesFM CE256 pretrain |
| `m1_monthly_dataset` | TimesFM CE256 pretrain |
| `weather_dataset` | TimesFM original head / CE64 / CE256 / staged pretrain |
| `electricity_hourly_dataset` | TimesFM original head / CE64 / CE256 / staged pretrain |
| `traffic_hourly_dataset` | TimesFM original head / CE64 / CE256 / staged pretrain |
| `temperature_rain_dataset_without_missing_values` | TimesFM staged starter / large pretrain |
| `kaggle_web_traffic_dataset_without_missing_values` | TimesFM staged large pretrain |
| `rideshare_dataset_without_missing_values` | TimesFM staged large pretrain |
| `vehicle_trips_dataset_without_missing_values` | TimesFM staged large pretrain |
| `kdd_cup_2018_dataset_without_missing_values` | TimesFM staged large pretrain |
| `covid_deaths_dataset` | TimesFM staged large pretrain |
| `nn5_daily_dataset_without_missing_values` | TimesFM staged large pretrain |
| `fred_md_dataset` | TimesFM staged large pretrain |

## 缺数据时的处理

如果运行 notebook 或脚本时报 `FileNotFoundError` / `Dataset not found`，检查对应文件是否仍在：

```text
data/extracted/<dataset>.tsf
```

如果需要恢复已删除的非主线数据集，请重新从 [Monash Time Series Forecasting Archive](https://forecastingdata.org/) 下载并解压到 `data/extracted/`。

Chronos 会复用 `data/arrow_training/<dataset>/` 下的 Arrow 文件；如果 `.tsf` 变动或怀疑缓存不一致，可以删除对应 Arrow 子目录后重跑。
