"""
Run gRNA optimization using a trained PPO model.

By default, off-target scoring is enabled (reference: data/genes_with_flankers.fna).
Disable with: --no-off-target.

Usage:
    python RL/run_optimize_grna.py --model RL/models/grna_ppo_500seqs_dense_ent005 --input data/run_normal_top50_sgRNA_with_crisprbert.csv --output data/optimized_sgrna.csv
    python RL/run_optimize_grna.py --model RL/models/grna_ppo_500seqs_dense_ent005 --input data/...csv --output data/...csv --reference-fasta data/genes_with_flankers.fna
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


def load_sequences_from_csv(csv_path: str, max_count: int = 10000):
    """Load 20-mer gRNA sequences from CSV (column: sgRNA)."""
    sequences = []
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        sg_col = next((h for h in fieldnames if h.strip().lower() == "sgrna"), None)
        if not sg_col:
            raise ValueError(f"CSV does not contain an 'sgRNA' column. Headers: {fieldnames}")
        for row in reader:
            s = (row.get(sg_col) or "").strip().upper().replace("U", "T")
            if len(s) != 20 or not all(c in "ATGC" for c in s):
                continue
            sequences.append(s)
            rows.append(row)
            if len(sequences) >= max_count:
                break
    return sequences, rows, fieldnames


def resolve_model_path(path: str, repo_root: str):
    """Resolve a model .zip path from file/directory input.
    If not found, also try RL/models/<name> and RL/models/<name>.zip.
    """
    path = os.path.abspath(path)
    name = os.path.basename(path.rstrip(os.sep)).replace(".zip", "")

    def try_path(p):
        if not p or not os.path.exists(p):
            return None
        if os.path.isfile(p):
            return p
        if os.path.isdir(p):
            zip_sibling = os.path.join(os.path.dirname(p), os.path.basename(p.rstrip(os.sep)) + ".zip")
            if os.path.isfile(zip_sibling):
                return zip_sibling
            zip_inside = os.path.join(p, name + ".zip")
            if os.path.isfile(zip_inside):
                return zip_inside
            zip_inside = os.path.join(p, "model.zip")
            if os.path.isfile(zip_inside):
                return zip_inside
            for f in os.listdir(p):
                if f.endswith(".zip"):
                    return os.path.join(p, f)
        if not p.endswith(".zip") and os.path.isfile(p + ".zip"):
            return p + ".zip"
        return None

    found = try_path(path)
    if found:
        return found
    if not path.endswith(".zip") and os.path.isfile(path + ".zip"):
        return path + ".zip"

    # Fallback: training usually saves models in RL/models/
    si_models = os.path.join(repo_root, "RL", "models")
    for candidate in [
        os.path.join(si_models, name + ".zip"),
        os.path.join(si_models, name),
    ]:
        found = try_path(candidate)
        if found:
            return found

    return path  # Return original path to improve the error message context


def main():
    parser = argparse.ArgumentParser(description="Optimize gRNA sequences with a trained PPO model")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to model (directory or .zip)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input CSV with an sgRNA column")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV (default: input filename with _optimized suffix)")
    parser.add_argument("--reference-fasta", type=str, default="data/genes_with_flankers.fna",
                        help="Path to reference FASTA (relative to repo root with RL/ and data/). Default: data/genes_with_flankers.fna")
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
        print(f"  --model RL/models/grna_ppo_500seqs_dense_ent005")
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
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    from RL.grna_gym_env import CRISPRGymEnv
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
            # Try common location: data/ under repo root
            fallback = os.path.join(_REPO_ROOT, "data", os.path.basename(ref_fasta))
            if os.path.isfile(fallback):
                ref_fasta = fallback
            else:
                fallback = os.path.join(_REPO_ROOT, "data", "genes_with_flankers.fna")
                if os.path.isfile(fallback):
                    ref_fasta = fallback
        if not os.path.isfile(ref_fasta):
            print(f"Warning: reference FASTA not found: {ref_fasta}")
            print("  Check the path. Typical location: data/genes_with_flankers.fna in the repository root.")
            print("  Off-target scoring will be disabled. Provide --reference-fasta to a .fna file or use --no-off-target.")
            ref_fasta = None
            use_crisprspec = False
    genome_seq = None
    if ref_fasta:
        print(f"Off-target: enabled, reference: {ref_fasta}")
        # Load the genome once and reuse the same object for every sequence, instead
        # of re-reading the FASTA (and re-transferring to GPU) on each iteration.
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

    # Load model
    model = PPO.load(model_path, device=device)
    print("Model loaded.")

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

    # Write results
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
