#!/usr/bin/env python3
"""
Run Doench + off-target/CFD + CRISPR-BERT metrics for
data/baseline_random_ppo_matched.csv.

Off-target search FASTA defaults to:
  data/genes_with_flankers.fna

Input CSV is expected to contain:
  - sgRNA_initial
  - sgRNA_random_best   (optimized random guide)
  - pam
  - window
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute metrics for baseline_random_ppo_matched.csv"
    )
    p.add_argument(
        "--input",
        default="data/baseline_random_ppo_matched.csv",
        help="Input CSV with sgRNA_initial / sgRNA_random_best.",
    )
    p.add_argument(
        "--genome",
        default="data/genes_with_flankers.fna",
        help="FASTA for off-target search.",
    )
    p.add_argument(
        "--output",
        default="data/baseline_random_ppo_matched_metrics_doench_offtarget_cfd_crisprbert.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--max-mismatches",
        type=int,
        default=4,
        help="Max mismatches for off-target search.",
    )
    p.add_argument(
        "--crisprbert-max-offtargets-per-guide",
        type=int,
        default=2000,
        help="Cap off-target pairs per guide for CRISPR-BERT.",
    )
    p.add_argument(
        "start_row",
        nargs="?",
        type=int,
        default=None,
        help="Optional first data row (1-based).",
    )
    p.add_argument(
        "end_row",
        nargs="?",
        type=int,
        default=None,
        help="Optional last data row (1-based).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    input_csv = (root / args.input).resolve()
    genome_fasta = (root / args.genome).resolve()
    output_csv = (root / args.output).resolve()

    cmd = [
        sys.executable,
        str((root / "compute_baseline_random_doench_offtarget_cfd_crisprbert.py").resolve()),
        "--grna-file",
        str(input_csv),
        "--offtarget-fasta",
        str(genome_fasta),
        "--output",
        str(output_csv),
        "--max-mismatches",
        str(args.max_mismatches),
        "--crisprbert-max-offtargets-per-guide",
        str(args.crisprbert_max_offtargets_per_guide),
    ]
    if args.start_row is not None:
        cmd.append(str(args.start_row))
    if args.end_row is not None:
        cmd.append(str(args.end_row))

    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[done] Saved: {output_csv}")


if __name__ == "__main__":
    main()

