# Waveform_CXR_EHR

当前仓库主入口是 **ECG-CXR patient-temporal contrastive learning** 实验。

主实验位于 [`ECGCXRPatientTemporal/`](ECGCXRPatientTemporal/)，目标是在冻结的
CXR / ECG latent space 中，检验 ECG 或 ECG 序列是否能对齐并预测同一患者未来
CXR embedding。详细设计、数据构建、loss、评估指标和 staged experiments 见：

- [`ECGCXRPatientTemporal/README.md`](ECGCXRPatientTemporal/README.md)

## 仓库结构

```text
Waveform_CXR_EHR/
├── ECGCXRPatientTemporal/      # 当前 contrastive learning 实验
├── data/                       # 当前实验默认读取的 CXR/ECG catalogs 与数据脚本
├── docs/archive/               # 历史文档
└── legacy/                     # 旧 baseline / transformer / artifact 代码归档
```

当前维护入口只有 `ECGCXRPatientTemporal/`。旧实验代码没有删除，统一收在
[`legacy/`](legacy/) 里，方便追溯但不再作为新人上手的主路径。

## 当前实验

核心问题：

- 单条 ECG 能否预测 9-15 小时后的 CXR latent state
- ECG 序列能否改善 future CXR retrieval
- 在 `CXR_t1 + ECG sequence -> CXR_t2` fusion 中，ECG 是否提供超出 `CXR_t1` shortcut 的信息

`run_staged.sh` 默认按 `run_experiments.py` 的 `ALL_IN_ORDER` 运行 Exp 1-4。
Exp 5/6 已在 registry 中实现，但属于显式 opt-in follow-up。

| CLI group | Experiment | Purpose |
|-----------|------------|---------|
| `step1` | Exp 1A/1B: single ECG -> future CXR | 测 ECG 单点信号；cross-only 与 combined loss |
| `step2` | Exp 2: single ECG + delta time -> future CXR | 加入 predictor `g` 与时间条件 |
| `step3` | Exp 3A/3B: ECG sequence -> CXR_t2 | 测 ECG temporal signal 与 future query |
| `step4` | Exp 4: CXR_t1 + ECG sequence -> CXR_t2 | 目标 fusion 模型与 ECG shortcut controls |
| `fusion_schemes` | Exp 5A/5B/5C | cross-attention、add+norm、weighted pooling fusion follow-ups |
| `improve` | Exp 5C + Exp 6A | weighted pooling 与 gated ECG residual follow-up |

## 快速入口

```bash
# 环境
source ECGCXRPatientTemporal/setup_env.sh

# 构建 ECG-CXR temporal pairs
bash ECGCXRPatientTemporal/jobs/run_build_pairs.sh
bash ECGCXRPatientTemporal/jobs/run_build_single_pairs.sh

# 预计算冻结 encoder embeddings。全套 staged 实验需要三个 pairs 文件的 union。
sbatch ECGCXRPatientTemporal/jobs/run_precompute.sh \
  --pairs ECGCXRPatientTemporal/artifacts/cache/default/patient_temporal_pairs.json \
          ECGCXRPatientTemporal/artifacts/cache/default/seq_target_pairs.json \
          ECGCXRPatientTemporal/artifacts/cache/default/single_ecg_pairs.json \
  --merge

# 运行默认 staged contrastive experiments（Exp 1-4）
sbatch ECGCXRPatientTemporal/jobs/run_staged.sh

# 可选 follow-up groups
sbatch ECGCXRPatientTemporal/jobs/run_staged.sh --only fusion_schemes
sbatch ECGCXRPatientTemporal/jobs/run_staged.sh --only improve
```

如需 full non-EHR-restricted catalogs、single-ECG pairs、cache merge 或 subset
实验命令，请直接看 [`ECGCXRPatientTemporal/README.md`](ECGCXRPatientTemporal/README.md)。

## 历史文档

旧的 EHR/CXR/ECG 分类、window transformer、baseline pipeline 说明已经归档到：

- [`docs/archive/`](docs/archive/)
- [`legacy/`](legacy/)

这些文档保留用于追溯早期实验，不再作为当前仓库入口。
