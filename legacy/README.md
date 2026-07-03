# Legacy Code Archive

这里集中保存当前 contrastive learning 实验之前的旧代码、图表和训练产物。

当前维护入口是 [`../ECGCXRPatientTemporal/`](../ECGCXRPatientTemporal/)。
本目录下代码主要用于追溯和复现实验历史，不建议新人从这里开始改。

## Layout

| Path | Content |
|------|---------|
| [`baselines/BaselineExperiment/`](baselines/BaselineExperiment/) | 旧 EHR / CXR / ECG / multimodal ARDS classification baselines |
| [`encoder_transformers/`](encoder_transformers/) | 旧 CXR、ECG、EHR encoder + Transformer 分类实验 |
| [`ehr_temporal/`](ehr_temporal/) | 旧 EHR trend / window transformer 实验 |
| [`early_matching/experiment1_old/`](early_matching/experiment1_old/) | 更早期 CXR-supertable-waveform matching pipeline |
| [`shared/models/`](shared/models/) | 旧实验共享 encoder 代码 |
| [`artifacts/`](artifacts/) | 旧 figures、logs、top-level output 与 Slurm 输出 |

## Notes

- 旧代码目录被整体搬迁后，部分脚本里的相对路径或 Slurm 命令可能仍反映历史位置。
- 如果需要复跑旧实验，请先从对应的 archived README 看当时设计，再按新目录位置调整运行路径。
- 当前 contrastive pipeline 已经内聚了自己需要的 CXR path / ECG loading helpers，不再依赖这里的 baseline code。
