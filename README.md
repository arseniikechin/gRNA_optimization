#gRNA-Optimization

> **Thermodynamically guided CRISPR spacer optimization: strategy-dependent off-target trade-offs**

---

## Overview

This repository contains the full implementation accompanying the paper. The agent iteratively mutates guide RNA sequences and is rewarded by a composite score:

- **CRISPRspec surrogate** — off-target specificity via thermodynamic energy modelling
- **GC-content preference** — bias towards 40–60% GC
- **Homopolymer penalty** — penalises runs of identical nucleotides

Inference and evaluation scripts are included for reproducible benchmarking.

---

## Repository Layout

```
RL/
├── train_grna_rl.py          # Main training script (PPO)
├── run_optimize_grna.py      # Apply trained model to CSV guides
├── analyze_policy.py         # Policy analysis and statistics
├── grna_gym_env.py           # Gym environment and reward logic
└── metrics/
    └── compute_eval_100_doench_offtarget_cfd_crisprbert.py  # Benchmark metrics
data/                         # Input and reference datasets
energy/                       # CRISPRspec energy pipeline resources
```

---

## Environment Setup

Requires **Python 3.10+** (3.10 or 3.11 recommended).

```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Notes:**
> - `rpy2` requires a working local [R installation](https://www.r-project.org/).
> - GPU is optional — all scripts support CPU via `--force-cpu` or `--device cpu`.

---

## Quick Start

All scripts should be run from the directory indicated in each section.

### 1. Train a new policy

*Run from `RL/`*

```bash
# Windows
python train_grna_rl.py ^
  --sequences ../data/train_400.txt ^
  --steps 200000 ^
  --seed-len 8 ^
  --max-episode-steps 20 ^
  --n-envs 4 ^
  --use-crisprspec ^
  --reference-fasta ../data/chr_3_7_20_21.fna

# Linux/macOS
python train_grna_rl.py \
  --sequences ../data/train_400.txt \
  --steps 200000 \
  --seed-len 8 \
  --max-episode-steps 20 \
  --n-envs 4 \
  --use-crisprspec \
  --reference-fasta ../data/chr_3_7_20_21.fna
```

Model saved to: `RL/models/grna_ppo.zip`

---

### 2. Optimise guides from CSV

Input CSV must contain an `sgRNA` column with valid 20-nt sequences.

*Run from `RL/`*

```bash
# Windows
python run_optimize_grna.py ^
  --model models/grna_ppo.zip ^
  --input ../data/run_normal_top50_sgRNA_with_crisprbert.csv ^
  --output ../data/optimized_sgrna.csv ^
  --reference-fasta ../data/genes_with_flankers.fna

# Linux/macOS
python run_optimize_grna.py \
  --model models/grna_ppo.zip \
  --input ../data/run_normal_top50_sgRNA_with_crisprbert.csv \
  --output ../data/optimized_sgrna.csv \
  --reference-fasta ../data/genes_with_flankers.fna
```

---

### 3. Analyse trained policy

*Run from `RL/`*

```bash
# Windows
python analyze_policy.py ^
  --model models/grna_ppo.zip ^
  --sequences ../data/eval_100.txt ^
  --genome ../data/chr_3_7_20_21.fna ^
  --output analysis_results

# Linux/macOS
python analyze_policy.py \
  --model models/grna_ppo.zip \
  --sequences ../data/eval_100.txt \
  --genome ../data/chr_3_7_20_21.fna \
  --output analysis_results
```

Outputs:
- `analysis_results/policy_report.txt`
- `analysis_results/policy_summary.json`

---

### 4. Compute benchmark metrics

*Run from `RL/metrics/`*

```bash
# Windows
python compute_eval_100_doench_offtarget_cfd_crisprbert.py ^
  --input ../data/eval_100_optimized.csv ^
  --output ../data/eval_100_optimized_metrics.csv ^
  --genome ../data/genes_with_flankers.fna

# Linux/macOS
python compute_eval_100_doench_offtarget_cfd_crisprbert.py \
  --input ../data/eval_100_optimized.csv \
  --output ../data/eval_100_optimized_metrics.csv \
  --genome ../data/genes_with_flankers.fna
```

---

## Reproducibility

To ensure reproducibility across experiments:

- Keep `--seed-len`, `--max-mismatches`, and reward weights identical between training and inference.
- Use the same reference FASTA for both training and evaluation whenever possible.
- Save run metadata (`logs/`, `training_summary.json`, full CLI commands) for each experiment.
- Record the exact model checkpoint (`.zip`) used for inference and policy analysis.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `stable-baselines3 not installed` | `pip install stable-baselines3[extra]` |
| CRISPRspec scoring disabled | Check FASTA path; pass `--reference-fasta data/genes_with_flankers.fna` |
| CSV load error in optimisation | Ensure `sgRNA` column exists and all sequences are valid 20-mers (`A/C/G/T`) |
| `rpy2` import error | Install R from [r-project.org](https://www.r-project.org/) and reinstall `rpy2` |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{,
  title   = {A reinforcement learning agent discovers thermodynamically grounded mutation rules for CRISPR guide RNA optimisation},
  author  = {},
  journal = {},
  year    = {2025},
  doi     = {}
}
```

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
