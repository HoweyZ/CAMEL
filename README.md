# CAMEL: Cross-temporal Anchor-guided Memory with Evolving Latent Dynamics

Official implementation for ultra-long-span traffic forecasting with temporal gaps.

## 1. Overview
Traffic forecasting under multi-year observation gaps is fundamentally different from standard short-horizon forecasting. CAMEL is designed for this setting and integrates three complementary components:
- **Cross-Year Episodic Memory (CEM):** retrieves season-consistent cross-year patterns as anchors.
- **Latent Dynamics Extrapolator (LDE):** propagates latent states through unobserved gaps via time-conditioned Graph-ODE dynamics.
- **Anchor-Temporal Fusion (ATF):** adaptively fuses memory, latent dynamics, and recent observations with gap-aware gating.

The method targets the **gap forecasting** protocol on long-span traffic benchmarks (e.g., XXLTraffic subsets such as PeMS03 / PeMS08 / TfNSW).


## 2. Repository Structure
```text
CAMEL/
├── run.py                          # training/testing entry
├── run.sh                          # batch script (switch dataset/gap via DATA_FILE / GAP_DAYS)
├── exp/
│   ├── exp_basic.py               # experiment base
│   └── exp_long_term_forecasting.py
├── logs/                           # training logs used in the paper (main table, ablation, strategies, etc.)
├── models/
│   ├── CAMEL.py                   # CAMEL core model
│   ├── Autoformer.py
│   ├── DLinear.py
│   ├── FEDformer.py
│   ├── Informer.py
│   ├── PatchTST.py
│   ├── iTransformer.py
│   ├── PhaseFormer.py
│   ├── MixLinear.py
│   ├── FreqCycle.py
│   ├── stgcn.py
│   ├── astgcn.py
│   ├── gwn.py
│   └── pdformer.py
├── data_provider/
│   ├── data_factory.py
│   ├── data_loader.py
│   ├── uea.py
│   └── m4.py
├── layers/
└── utils/
```

## 3. Training 

### Batch run
```bash
bash run.sh
```

### Switch dataset or gap range
`run.sh` supports quick switching of dataset file and gap span:

```bash
# example: run on PeMS03 with 1.5y and 2y gaps
DATA_FILE=pems03_all_common_flow.csv GAP_DAYS="548 730" bash run.sh CAMEL
```

You can also set:
- `DATA_ROOT` to change dataset directory.
- `DATASET_TAG` to override experiment name prefix in `model_id`.

## 4. Supported Baselines in This Repo
- MLP-based: `DLinear`, `FreqCycle`, `MixLinear`
- Transformer-based: `Informer`, `Autoformer`, `FEDformer`, `PatchTST`, `iTransformer`, `PhaseFormer`
- Graph-based: `stgcn`, `astgcn`, `gwn`, `pdformer`
- Proposed method: `CAMEL`

# CAMEL
