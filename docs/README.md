# Docs

本目录只保留当前实验结论需要的说明文件。

- [CHRONOS.md](CHRONOS.md)：Chronos-T5 的损失函数改造、实现细节、注意事项、结果与讨论。
- [TIMESFM.md](TIMESFM.md)：TimesFM 2.5 的回归基线、CE head 改造、实现细节、注意事项、结果与讨论。

通用注意事项：

- 推荐使用 [notebook/REPRODUCE_RESULTS.ipynb](../notebook/REPRODUCE_RESULTS.ipynb) 复现当前结果。
- 如果只需要加载已有 checkpoint 做 eval 和预测展示，使用 [notebook/EVALUATE_RESULTS.ipynb](../notebook/EVALUATE_RESULTS.ipynb)。
- 基座模型权重不在仓库中，Chronos 和 TimesFM 都依赖 Hugging Face cache 或联网下载。
- 数据来自 [Monash Time Series Forecasting Archive](https://forecastingdata.org/)，默认读取 `data/extracted/*.tsf`；缺数据时普通脚本一般会抛 `FileNotFoundError`。
- Chronos 与 TimesFM 的 tourism_monthly 切分口径不同，不建议把两者数值做严格横向比较。
- 当前保留的结果目录在 `output/` 下；notebook 会把新复现结果写入 `output/notebook_repro_时间戳/`。
