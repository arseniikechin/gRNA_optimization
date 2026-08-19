# Thermodynamically guided CRISPR spacer optimization: strategy-dependent off-target trade-offs

Code and data for the paper: a controlled comparison of **five** search strategies that refine 20-nt SpCas9 spacers under one thermodynamically grounded objective. The PAM-proximal seed is fixed; at most four substitutions are allowed in the PAM-distal region.

All five methods maximize the same composite score

$$S(s) = w_{\mathrm{spec}}\,f_{\mathrm{spec}}(s) - w_{\mathrm{gc}}\,p_{\mathrm{gc}}(s) - w_{\mathrm{hp}}\,p_{\mathrm{hp}}(s)$$

with $w_{\mathrm{spec}}=1.0$ and $w_{\mathrm{gc}}=w_{\mathrm{hp}}=0.1$:

- **CRISPRspec** ($f_{\mathrm{spec}}$) — off-target specificity from RNA:DNA hybridization energies (`energy/`)
- **GC-content** ($p_{\mathrm{gc}}$) — penalty for deviation from 50% GC (prefer 40–60%)
- **Homopolymer** ($p_{\mathrm{hp}}$) — penalty for runs longer than three identical bases

| Strategy | Role in the paper | Location |
|---|---|---|
| PPO | on-policy RL policy | `RL/train_grna_rl.py`, `RL/models/grna_ppo_best.zip` |
| QR-DQN | distributional off-policy RL | `RL/train_grna_qrdqn.py`, `RL/models/grna_qrdqn_best.zip` |
| Rainbow DQN | Rainbow (PyTorch) | `RL/train_grna_rainbow.py`, `RL/models/grna_rainbow.zip` |
| Score-guided search | non-learning baseline: 20 random mutants, keep $\arg\max S$; edit count matched to PPO | `Score-guided search/` |
| Iterative LLM | prompt-guided `openai/gpt-oss-120b`, no fine-tuning | `LLM/` |

---

## Environment (shared by all strategies)

| | |
|---|---|
| State | one-hot spacer, shape `(20, 4)` |
| Action | position × nucleotide in the mutable region (`Discrete(48)` at `seed_len=8`) |
| Mutable | positions 1–12 (0-based 0–11); seed positions 13–20 stay fixed |
| Budget | Hamming distance $\le 4$ from the initial spacer; $\le 20$ steps |
| Sequential reward | $r_t = S(s_{t+1}) - S(s_t)$ (dense) |

Invalid actions (same base, or over the mismatch budget) are no-ops. A terminal penalty applies if an episode uses fewer than the minimum number of substitutions.

---

## Repository layout

```
RL/
├── train_grna_rl.py              # PPO (stable-baselines3)
├── train_grna_qrdqn.py           # QR-DQN (sb3-contrib)
├── train_grna_rainbow.py         # Rainbow DQN (PyTorch)
├── run_optimize_grna.py          # PPO inference on a CSV
├── run_optimize_grna_qrdqn.py    # QR-DQN inference
├── run_optimize_grna_rainbow.py  # Rainbow inference (.pt checkpoints)
├── analyze_policy.py             # PPO / QR-DQN / Rainbow analysis (auto-detect)
├── grna_gym_env.py               # Gymnasium env + composite score
├── grna_rl_adapters.py           # genome load, CPU off-target search, CRISPRspec
├── grna_rl_adapters_cuda.py      # optional CUDA off-target search
├── models/                       # trained policies from the paper
└── metrics/                      # Doench / CFD / CRISPR-BERT post-hoc metrics
Score-guided search/
├── baseline_random_ppo_matched.py
└── run_baseline_random_ppo_matched_metrics.py
LLM/
└── llm.meta.json                 # prompts, config, and metrics of the paper LLM run
data/
├── train_400.txt                 # 400 training 20-mers (GHR, PRLR, IGF1R, INSR, LEPR)
├── eval_100.txt                  # 100 held-out IGF2R 20-mers (chromosome 9)
├── chr_3_7_20_21.fna             # chromosomes 3, 7, 20, 21 (~390 Mbp, Git LFS)
└── genes_with_flankers.fna       # locus reference: gene bodies ±1 Mb (Git LFS)
energy/                           # CRISPRspec / CRISPRoff energy tables
```

Run scripts from the **repository root**. RL modules import `RL.*`.

**Data.** Bos taurus ARS-UCD2.0 (RefSeq GCF_002263795.3). Training used 400 diverse NGG spacers from five endocrine-receptor genes; evaluation used 100 IGF2R guides on a chromosome excluded from training. Optimization scored off-targets on `chr_3_7_20_21.fna`.

---

## Setup

Python **3.10** (3.10 or 3.11). The `.fna` files are Git LFS pointers — pull them before training or scoring:

```bash
git lfs install
git lfs pull
```

### Option A — conda (Linux + CUDA)

`requirements.txt` is a **conda environment export** (linux-64, CUDA 12.x), not a pip freeze:

```bash
conda create --name grna --file requirements.txt
conda activate grna
```

This will not work as-is on macOS or Windows.

### Option B — pip (portable)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install "numpy" "pandas" "biopython" "gymnasium" \
  "stable-baselines3[extra]" "sb3-contrib" "torch" "tqdm" "numba"
```

Optional:

- GPU off-target search: CUDA + Numba CUDA (used automatically when available).
- CRISPRspec self-folding / RNAfold metrics: [ViennaRNA](https://www.tbi.univie.ac.at/RNA/).
- Doench 2016: local [R](https://www.r-project.org/) and `rpy2`.
- CRISPR-BERT: TensorFlow / Keras and weights under `RL/metrics/CRISPR_BERT/`.
- LLM baseline: an OpenRouter (or compatible) API key for `openai/gpt-oss-120b`.

CPU is always supported: `--force-cpu` (training) or `--device cpu` (inference).

---

## 1. Reinforcement learning (PPO, QR-DQN, Rainbow)

The paper trained each policy for $10^6$ environment steps on `data/train_400.txt`. Checkpoints from that comparison are in `RL/models/`.

### Use a trained policy

Inference CSVs need an `sgRNA` column of valid 20-mers (`A/C/G/T`). From `eval_100.txt`:

```bash
python -c "
from pathlib import Path
seqs = [l.strip() for l in Path('data/eval_100.txt').read_text().splitlines() if l.strip()]
Path('data/eval_100.csv').write_text('sgRNA\n' + '\n'.join(seqs) + '\n')
"
```

**PPO**

```bash
python RL/run_optimize_grna.py \
  --model RL/models/grna_ppo_best.zip \
  --input data/eval_100.csv \
  --output data/eval_100_optimized_ppo.csv \
  --reference-fasta data/genes_with_flankers.fna \
  --seed-len 8
```

**QR-DQN**

```bash
python RL/run_optimize_grna_qrdqn.py \
  --model RL/models/grna_qrdqn_best.zip \
  --input data/eval_100.csv \
  --output data/eval_100_optimized_qrdqn.csv \
  --reference-fasta data/genes_with_flankers.fna \
  --seed-len 8
```

Output columns: original fields plus `sgRNA_optimized`, `score_initial`, `score_final`, `steps_used`.

`--seed-len` and `--max-mismatches` must match training (defaults 8 and 4). Pass `--no-off-target` to skip CRISPRspec.

**Rainbow.** `analyze_policy.py` loads `RL/models/grna_rainbow.zip`. `run_optimize_grna_rainbow.py` expects a `.pt` file written by `train_grna_rainbow.py`.

### Train

Paper protocol: `--steps 1000000`, `--seed-len 8`, `--max-episode-steps 20`, `--use-crisprspec`, `--reference-fasta data/chr_3_7_20_21.fna`. Shorter runs work for debugging.

**PPO** → `RL/models/grna_ppo.zip`

```bash
python RL/train_grna_rl.py \
  --sequences data/train_400.txt \
  --eval-sequences data/eval_100.txt \
  --steps 1000000 \
  --seed-len 8 \
  --max-episode-steps 20 \
  --n-envs 4 \
  --use-crisprspec \
  --reference-fasta data/chr_3_7_20_21.fna
```

**QR-DQN** → `RL/models/grna_qrdqn.zip` (default `--n-envs 1`)

```bash
python RL/train_grna_qrdqn.py \
  --sequences data/train_400.txt \
  --eval-sequences data/eval_100.txt \
  --steps 1000000 \
  --seed-len 8 \
  --max-episode-steps 20 \
  --use-crisprspec \
  --reference-fasta data/chr_3_7_20_21.fna
```

**Rainbow** → `RL/models/grna_rainbow.pt`

```bash
python RL/train_grna_rainbow.py \
  --sequences data/train_400.txt \
  --steps 1000000 \
  --seed-len 8 \
  --max-episode-steps 20 \
  --use-crisprspec \
  --reference-fasta data/chr_3_7_20_21.fna
```

Without `--use-crisprspec` and a readable FASTA, CRISPRspec is 0 and only GC / homopolymer terms remain.

### Analyse a trained policy

Works for PPO, QR-DQN, and Rainbow (checkpoint type is detected automatically):

```bash
python RL/analyze_policy.py \
  --model RL/models/grna_ppo_best.zip \
  --sequences data/eval_100.txt \
  --genome data/chr_3_7_20_21.fna \
  --output analysis_results
```

Writes `analysis_results/policy_report.txt` and `analysis_results/policy_summary.json` (positional and nucleotide-substitution preferences, as in Figure 3 of the paper).

---

## 2. Score-guided search

Non-learning baseline from the paper (Appendix A.4). It does **not** train a policy. For each input guide it:

1. Reads the PPO-optimized spacer and sets $k = \min(\mathrm{Hamming}(\mathrm{sgRNA}, \mathrm{sgRNA\_optimized}), 4)$.
2. Draws **20** random mutants that each apply exactly $k$ substitutions in the mutable region (positions 0–11).
3. Scores every mutant with the same $S$ (CRISPRspec + GC + homopolymer).
4. Keeps the highest-scoring variant.

Matching $k$ to PPO isolates the effect of *where* substitutions are placed, not how many.

Input CSV must contain `sgRNA` (initial 20-mer) and `sgRNA_optimized` (PPO output from §1).

```bash
python "Score-guided search/baseline_random_ppo_matched.py" \
  --csv data/eval_100_optimized_ppo.csv \
  --fasta data/genes_with_flankers.fna \
  --out data/baseline_random_ppo_matched.csv \
  --repeats 20 \
  --seed-len 8 \
  --max-mismatches 4 \
  --seed 42
```

Output includes `sgRNA_initial`, `sgRNA_random_best`, scores, off-target counts, and `k_used`. `--no-offtarget` skips genomic scoring (debug only).

Post-hoc Doench / CFD / CRISPR-BERT for this baseline:

```bash
python "Score-guided search/run_baseline_random_ppo_matched_metrics.py" \
  --input data/baseline_random_ppo_matched.csv \
  --genome data/genes_with_flankers.fna \
  --output data/baseline_random_ppo_matched_metrics.csv
```

The metrics wrapper expects `sgRNA_initial`, `sgRNA_random_best`, `pam`, and `window`. Scoring helpers are imported as `baseline_random_v2` from the same directory as the baseline script.

---

## 3. Iterative LLM (`openai/gpt-oss-120b`)

The fifth strategy is a prompt-guided language model with **no task-specific fine-tuning**. It searches the same action space and score $S$ as the RL policies and the score-guided baseline.

**Protocol (paper, Appendix A.4)**

- Model: `openai/gpt-oss-120b` via OpenRouter (primary backend `google-vertex`; fallbacks `sambanova`, `groq`, `amazon-bedrock`, `digitalocean`).
- Temperature `0.7`; up to **20** iterations per guide; retain the highest-$S$ spacer seen.
- Each response must be a single JSON object: `{"spacer": "..."}`. Invalid JSON is retried up to three times.
- Constraints identical to the MDP: 20-mer over `{A,C,G,T}`, editable positions 0–11, seed 12–19 fixed, at most four substitutions.

At each iteration the prompt includes:

- current and best spacer so far
- seed-region and four-substitution constraints
- composite score and its CRISPRspec, GC, and homopolymer terms
- number of detected off-target sites
- the five highest-scoring and three most recent previous proposals
- design notes: keep GC in 40–60%, avoid long homopolymers, prefer G/C where useful, do not spend the full edit budget unless needed

`LLM/llm.meta.json` stores the exact prompts and hyperparameters of the reported run (`prompt_variant: gRNA-domain`, `aggressiveness: minimal`, `top_k: 5`, `recent_n: 3`, `seed_len: 8`, `max_mismatches: 4`, genome `data/chr_3_7_20_21.fna`) plus summary metrics on the 100 IGF2R guides.

To reproduce, call the same model with those prompts, score each proposed spacer with the shared CRISPRspec pipeline (`RL/grna_rl_adapters.py` + `energy/`), reject illegal edits, and keep $\arg\max S$ over the 20 iterations.

---

## Independent evaluation metrics

`RL/metrics/compute_eval_100_doench_offtarget_cfd_crisprbert.py` scores initial vs optimized guides with quantities that were **not** in $S$:

1. Doench 2016 / Rule Set 2 (needs R / `rpy2` and a 30-nt `window` column)
2. Genomic off-targets (≤ `--max-mismatches`, PAM NGG) + CFD
3. CRISPR-BERT over off-target pairs

Required CSV columns: `sgRNA`, `sgRNA_optimized`; also `pam` and `window` for Doench.

```bash
python RL/metrics/compute_eval_100_doench_offtarget_cfd_crisprbert.py \
  --input data/eval_100_optimized_ppo.csv \
  --output data/eval_100_optimized_metrics.csv \
  --genome data/genes_with_flankers.fna \
  --skip-doench
```

`--skip-doench`, `--skip-offtarget-cfd`, and `--skip-crisprbert` turn individual blocks off.

---

## Reproducibility

- Keep `--seed-len`, `--max-mismatches`, and reward weights identical across every strategy.
- The paper optimized against `data/chr_3_7_20_21.fna` and reported the five-way comparison on the locus FASTA `data/genes_with_flankers.fna` (per-guide ΔOT correlated at $r=0.986$ with the four-chromosome reference).
- Score-guided search must see a PPO CSV so that $k$ is matched per guide.
- LLM results in the paper are from a single run; record backend, temperature, and `llm.meta.json`.
- Save `logs/`, `training_summary.json`, and the exact checkpoint (`.zip` or `.pt`).
- Record whether off-target search ran on CUDA or CPU.

---

## Troubleshooting

| Issue | What to do |
|---|---|
| LFS pointer instead of FASTA (`version https://git-lfs.github.com/...`) | `git lfs pull` |
| `stable-baselines3 not installed` | `pip install "stable-baselines3[extra]"` |
| `sb3-contrib` / QR-DQN import error | `pip install sb3-contrib` |
| CRISPRspec stays 0 | pass `--use-crisprspec` and a real FASTA via `--reference-fasta` |
| CSV load error | column `sgRNA`, 20-mers over `A/C/G/T` only |
| Score-guided: `No module named baseline_random_v2` | put the shared scoring helper next to `baseline_random_ppo_matched.py` |
| `pip install -r requirements.txt` fails | that file is a conda export; use Option A or the pip list in Option B |
| `rpy2` / Doench error | install R, then `pip install rpy2`; or pass `--skip-doench` |

---

## Citation

```bibtex
@article{kechin2026grna,
  title   = {Thermodynamically guided {CRISPR} spacer optimization: strategy-dependent off-target trade-offs},
  author  = {Kechin, Arsenii and Vepreva, Anastasia and Dubrovsky, Ivan and Reshetnyak, Polina and Dmitrenko, Andrei and Nikitin, Nikolay and Serov, Nikita},
  year    = {2026},
  note    = {ITMO University}
}
```

---

## License

MIT. See [`LICENSE`](LICENSE). The CRISPRspec pipeline under `energy/` is from CRISPRoff (GNU AGPL). CRISPR-BERT under `RL/metrics/CRISPR_BERT/` retains its upstream license.
