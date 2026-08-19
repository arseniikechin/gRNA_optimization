"""
Run gRNA optimization using a trained QR-DQN model (sb3-contrib).

This is the QR-DQN counterpart of run_optimize_grna.py: identical environment, inputs
and outputs, but the model is loaded with QRDQN.load instead of PPO.load.

By default, off-target scoring is enabled (reference: data/genes_with_flankers.fna).
Disable with: --no-off-target.

Usage:
    python RL/run_optimize_grna_qrdqn.py \
        --model RL/models/grna_qrdqn.zip \
        --input data/diverse_100_per_gene.csv \
        --output data/diverse_100_per_gene_optimized.csv \
        --reference-fasta data/genes_with_flankers.fna --seed-len 8
"""

import argparse
import csv
import os
import sys

# UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Repository root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse model-path resolution and CSV loading from the PPO inference script.
from RL.run_optimize_grna import resolve_model_path, load_sequences_from_csv


def main():
    parser = argparse.ArgumentParser(description="Optimize gRNA sequences with a trained QR-DQN model")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to QR-DQN model (directory or .zip)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input CSV with an sgRNA column")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV (default: input filename with _optimized suffix)")
    parser.add_argument("--reference-fasta", type=str, default="data/genes_with_flankers.fna",
                        help="Path to reference FASTA (relative to repo root). Default: data/genes_with_flankers.fna")
    parser.add_argument("--no-off-target", action="store_true",
                        help="Disable off-target scoring (ignore --reference-fasta)")
    parser.add_argument("--seed-len", type=int, default=8,
                        help="Seed region length near PAM (must match training). 8 -> 12 mutable positions, 48 actions")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Maximum optimization steps per sequence")
    parser.add_argument("--max-mismatches", type=int, default=4,
                        help="Maximum mismatches from original sequence")
    parser.add_argument("--device", type=str, default="auto",
                        choices=("auto", "cuda", "cpu"),
                        help="Model device (auto = cuda if available)")
    parser.add_argument("--max-sequences", type=int, default=0,
                        help="Maximum number of sequences to process (0 = all)")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model, _REPO_ROOT)
    if not os.path.isfile(model_path):
        basename = os.path.basename(args.model.rstrip(os.sep).replace(".zip", "")) + ".zip"
        si_default = os.path.join(_REPO_ROOT, "RL", "models", basename)
        print(f"Error: model not found: {model_path}")
        print("Hint: training typically saves models in RL/models/. Try:")
        print(f"  --model RL/models/grna_qrdqn")
        print("  or an absolute path to the .zip file, for example:")
        print(f"  --model {si_default}")
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    out_path = args.output
    if not out_path:
        base, ext = os.path.splitext(args.input)
        out_path = base + "_optimized" + ext

    sequences, rows, fieldnames = load_sequences_from_csv(
        args.input,
        max_count=args.max_sequences if args.max_sequences else 100000
    )
    if not sequences:
        print("No valid 20-mer sequences found in the sgRNA column.")
        sys.exit(1)

    print(f"Loaded sequences: {len(sequences)}")
    print(f"Model: {model_path}")
    print(f"Output: {out_path}")

    # Imports after path setup
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    try:
        from sb3_contrib import QRDQN
    except ImportError:
        print("Error: sb3-contrib (QR-DQN) not installed.")
        print("Install with: pip install sb3-contrib")
        sys.exit(1)

    from RL.train_grna_rl import make_env

    # Match training environment settings; use off-target if reference FASTA is available
    seed_len = args.seed_len
    max_steps = args.max_steps
    max_mismatches = args.max_mismatches
    min_mismatches = 1
    ref_fasta = None if getattr(args, "no_off_target", False) else args.reference_fasta
    use_crisprspec = bool(ref_fasta)
    if ref_fasta:
        if not os.path.isabs(ref_fasta):
            ref_fasta = os.path.normpath(os.path.join(_REPO_ROOT, ref_fasta))
        if not os.path.isfile(ref_fasta):
            fallback = os.path.join(_REPO_ROOT, "data", os.path.basename(ref_fasta))
            if os.path.isfile(fallback):
                ref_fasta = fallback
            else:
                fallback = os.path.join(_REPO_ROOT, "data", "genes_with_flankers.fna")
                if os.path.isfile(fallback):
                    ref_fasta = fallback
        if not os.path.isfile(ref_fasta):
            print(f"Warning: reference FASTA not found: {ref_fasta}")
            print("  Off-target scoring will be disabled. Provide --reference-fasta or use --no-off-target.")
            ref_fasta = None
            use_crisprspec = False

    genome_seq = None
    if ref_fasta:
        print(f"Off-target: enabled, reference: {ref_fasta}")
        # Load the genome once and reuse the same object for every sequence.
        from RL.grna_rl_adapters import load_genome
        genome_seq = load_genome(ref_fasta)
        print(f"Genome loaded once: {len(genome_seq)} bp")
        ref_fasta = None  # pass genome_seq directly to the env
    elif use_crisprspec is False and not getattr(args, "no_off_target", False):
        print("Off-target: disabled (reference FASTA not found or --no-off-target)")

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Device: {device}")

    # Load QR-DQN model
    model = QRDQN.load(model_path, device=device)
    print("Model loaded (QR-DQN).")

    # Environment consistent with training: CRISPRspec + GC + homopolymer
    crisprspec_weight = 1.0
    gc_weight = 0.1
    homopolymer_weight = 0.1

    def make_single_env(initial_seqs):
        return make_env(
            initial_seqs,
            seed_len,
            max_steps,
            max_mismatches,
            min_mismatches,
            use_crisprspec=use_crisprspec,
            genome_seq=genome_seq,
            crisprspec_weight=crisprspec_weight,
            gc_weight=gc_weight,
            homopolymer_weight=homopolymer_weight,
            use_cuda_offtarget=(device == "cuda"),
            use_dense_reward=False,
            rank=0,
            reference_fasta_path=ref_fasta,
        )

    results = []
    for i, (seq, row) in enumerate(zip(sequences, rows)):
        env = DummyVecEnv([lambda s=seq: Monitor(make_single_env([s])())])
        obs = env.reset()
        done = False
        step = 0
        infos = [{}]
        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            done = dones[0]
            step += 1
        info = infos[0] if infos else {}
        optimized = info.get("sequence", seq)
        score = info.get("score", None)
        initial_score = info.get("initial_score", None)
        results.append({
            "row": row,
            "sgRNA_original": seq,
            "sgRNA_optimized": optimized,
            "score_final": score,
            "score_initial": initial_score,
            "steps_used": step,
        })
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed: {i + 1}/{len(sequences)}")

    env.close()

    out_fieldnames = list(fieldnames) + ["sgRNA_optimized", "score_initial", "score_final", "steps_used"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            out_row = dict(r["row"])
            out_row["sgRNA_optimized"] = r["sgRNA_optimized"]
            out_row["score_initial"] = r["score_initial"] if r["score_initial"] is not None else ""
            out_row["score_final"] = r["score_final"] if r["score_final"] is not None else ""
            out_row["steps_used"] = r["steps_used"]
            writer.writerow(out_row)

    print(f"Done. Results written to: {out_path}")


if __name__ == "__main__":
    main()
