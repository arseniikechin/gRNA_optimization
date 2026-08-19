#!/usr/bin/env python3
"""
Random baseline with PPO-matched mutation count per guide.

For each guide:
  - Mutation count k is derived from PPO result:
      k_ppo = Hamming(sgRNA, sgRNA_optimized)
      k_used = min(k_ppo, max_mismatches, mutable_region_size)
  - Each random trial for this guide uses exactly k_used substitutions.
  - Default --repeats is 20.

Input CSV must contain:
  - sgRNA (initial 20-mer)
  - sgRNA_optimized (PPO final 20-mer)
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import baseline_random_v2 as v2

SEQ_LEN = 20
BASES = ["A", "C", "G", "T"]


def random_mutant_fixed_k(
    sequence: str,
    mutable_positions: List[int],
    k: int,
    rng: random.Random,
) -> str:
    seq = list(sequence.upper())
    if k <= 0:
        return "".join(seq)

    positions = rng.sample(mutable_positions, k)
    for pos in positions:
        current = seq[pos]
        choices = [b for b in BASES if b != current]
        seq[pos] = rng.choice(choices)
    return "".join(seq)


def run_random_baseline_single_fixed_k(
    sequence: str,
    k_used: int,
    mutable_positions: List[int],
    score_fn,
    n_repeats: int,
    rng: random.Random,
    w_spec: float = 1.0,
    w_gc: float = 0.1,
    w_hp: float = 0.1,
) -> Dict:
    sequence = sequence.upper()

    n_ot_init, cs_init = score_fn(sequence)
    fspec_init = cs_init / 10.0
    gc_pen_init = v2.compute_gc_penalty(sequence)
    hp_pen_init = v2.compute_homopolymer_penalty(sequence)
    s_init = v2.composite_score(fspec_init, gc_pen_init, hp_pen_init, w_spec, w_gc, w_hp)

    n_ot_list, cs_list, s_list, seq_list = [], [], [], []
    for _ in range(n_repeats):
        mutant = random_mutant_fixed_k(sequence, mutable_positions, k_used, rng)
        n_ot, cs = score_fn(mutant)
        fspec = cs / 10.0
        gc_pen = v2.compute_gc_penalty(mutant)
        hp_pen = v2.compute_homopolymer_penalty(mutant)
        s = v2.composite_score(fspec, gc_pen, hp_pen, w_spec, w_gc, w_hp)
        n_ot_list.append(n_ot)
        cs_list.append(cs)
        s_list.append(s)
        seq_list.append(mutant)

    best_idx = int(np.argmax(s_list))
    return {
        "sgRNA_initial": sequence,
        "n_ot_initial": n_ot_init,
        "crisprspec_initial": round(cs_init, 4),
        "score_initial": round(s_init, 4),
        "sgRNA_random_best": seq_list[best_idx],
        "n_ot_random_best": n_ot_list[best_idx],
        "crisprspec_random_best": round(cs_list[best_idx], 4),
        "score_random_best": round(s_list[best_idx], 4),
        "n_mutations_best": v2.count_mismatches(sequence, seq_list[best_idx]),
        "n_ot_random_mean": round(float(np.mean(n_ot_list)), 2),
        "crisprspec_random_mean": round(float(np.mean(cs_list)), 4),
        "score_random_mean": round(float(np.mean(s_list)), 4),
        "delta_n_ot": n_ot_list[best_idx] - n_ot_init,
        "delta_score": round(s_list[best_idx] - s_init, 4),
        "n_repeats": n_repeats,
        "selection_criterion": "argmax_composite_score_fixed_k_from_ppo",
        "ppo_mutations_total": int(k_used),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Random baseline with per-guide mutation count matched to PPO."
    )
    p.add_argument("--csv", required=True, help="Input CSV with sgRNA and sgRNA_optimized.")
    p.add_argument("--fasta", required=True, help="Reference FASTA for off-target scoring.")
    p.add_argument("--out", default="baseline_random_ppo_matched.csv", help="Output CSV.")
    p.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Random trials per guide (default: 20).",
    )
    p.add_argument("--max-mismatches", type=int, default=4, help="Maximum allowed k (default: 4).")
    p.add_argument("--seed-len", type=int, default=8, help="Fixed seed length at 3' end.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--w-spec", type=float, default=1.0, help="CRISPRspec weight.")
    p.add_argument("--w-gc", type=float, default=0.1, help="GC penalty weight.")
    p.add_argument("--w-hp", type=float, default=0.1, help="Homopolymer penalty weight.")
    p.add_argument("--no-offtarget", action="store_true", help="Skip off-target scoring.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        sys.exit(f"ERROR: CSV not found: {csv_path}")
    if not args.no_offtarget and not os.path.isfile(args.fasta):
        sys.exit(f"ERROR: FASTA not found: {args.fasta}")

    df = pd.read_csv(csv_path)
    for col in ("sgRNA", "sgRNA_optimized"):
        if col not in df.columns:
            sys.exit(f"ERROR: Missing required column '{col}' in {csv_path}")

    mask = (df["sgRNA"].astype(str).str.len() == SEQ_LEN) & (df["sgRNA_optimized"].astype(str).str.len() == SEQ_LEN)
    df = df[mask].reset_index(drop=True)
    if df.empty:
        sys.exit("ERROR: No valid 20-mer rows found.")

    meta_cols = [c for c in df.columns if c not in ("sgRNA", "sgRNA_optimized")]
    mutable_len = SEQ_LEN - args.seed_len
    mutable_positions = list(range(mutable_len))

    if args.no_offtarget:
        print("[baseline_random_ppo_matched] --no-offtarget: scoring skipped.")
        score_fn = lambda g: (0, 10.0)
    else:
        score_fn = v2.load_offtarget_fn(args.fasta)

    rng = random.Random(args.seed)
    n_guides = len(df)
    logger = v2.ProgressLogger(n_total=n_guides, n_repeats=args.repeats)
    logger.start()

    print(f"  Input file      : {csv_path} ({n_guides} guides)")
    print(f"  Mutable region  : positions 0..{mutable_len - 1}")
    print(f"  k per guide     : Hamming(sgRNA, sgRNA_optimized), capped at {args.max_mismatches}")
    print(f"  Repeats/guide   : {args.repeats}")
    print(f"  Output          : {args.out}")
    print()

    results: List[Dict] = []

    t_probe_start = time.time()
    for i, row in df.iterrows():
        seq = str(row["sgRNA"]).upper()
        seq_ppo = str(row["sgRNA_optimized"]).upper()
        k_ppo = v2.count_mismatches(seq, seq_ppo)
        k_used = min(k_ppo, args.max_mismatches, len(mutable_positions))

        t0 = time.time()
        res = run_random_baseline_single_fixed_k(
            sequence=seq,
            k_used=k_used,
            mutable_positions=mutable_positions,
            score_fn=score_fn,
            n_repeats=args.repeats,
            rng=rng,
            w_spec=args.w_spec,
            w_gc=args.w_gc,
            w_hp=args.w_hp,
        )
        elapsed = time.time() - t0

        res["ppo_mutations_total"] = int(k_ppo)
        res["k_used"] = int(k_used)
        for col in meta_cols:
            res[col] = row[col]
        results.append(res)
        logger.update(i, seq, res, elapsed)

        if i == 0:
            probe = time.time() - t_probe_start
            est_total = probe * n_guides
            finish = datetime.now() + timedelta(seconds=est_total)
            print(f"  Guide 1 time    : {probe:.1f}s")
            print(f"  Est. total      : {v2._fmt_duration(est_total)} (finish ~{finish.strftime('%H:%M')})")
            print()

    logger.finish()

    out_df = pd.DataFrame(results)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out} ({len(out_df)} rows)")
    v2._print_summary(out_df)


if __name__ == "__main__":
    main()

