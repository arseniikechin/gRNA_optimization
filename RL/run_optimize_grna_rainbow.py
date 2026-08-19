"""
Run gRNA optimization using a trained Rainbow DQN model.

This is the Rainbow counterpart of run_optimize_grna.py. The Rainbow agent is a plain
PyTorch model (saved as a .pt checkpoint by train_grna_rainbow.py), so it is rebuilt
from the checkpoint config here rather than loaded via SB3.

Inputs/outputs match the PPO/QR-DQN inference scripts: an input CSV with an `sgRNA`
column, and an output CSV with `sgRNA_optimized`, `score_initial`, `score_final`,
`steps_used`.

Usage:
    python RL/run_optimize_grna_rainbow.py \
        --model RL/models/grna_rainbow.pt \
        --input data/diverse_100_per_gene.csv \
        --output data/diverse_100_per_gene_optimized_rainbow.csv \
        --reference-fasta data/genes_with_flankers.fna
"""

import argparse
import csv
import os
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from RL.run_optimize_grna import load_sequences_from_csv


def resolve_pt_path(path: str, repo_root: str) -> str:
    """Resolve a Rainbow .pt checkpoint path (accepts with/without extension)."""
    path = os.path.abspath(path)
    candidates = [path]
    if not path.endswith(".pt"):
        candidates.append(path + ".pt")
    name = os.path.basename(path.rstrip(os.sep)).replace(".pt", "")
    models_dir = os.path.join(repo_root, "RL", "models")
    candidates += [
        os.path.join(models_dir, name + ".pt"),
        os.path.join(models_dir, name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return path


def main():
    parser = argparse.ArgumentParser(description="Optimize gRNA sequences with a trained Rainbow DQN model")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to Rainbow checkpoint (.pt)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV with an sgRNA column")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV (default: input filename with _optimized suffix)")
    parser.add_argument("--reference-fasta", type=str, default="data/genes_with_flankers.fna",
                        help="Path to reference FASTA (relative to repo root). Default: data/genes_with_flankers.fna")
    parser.add_argument("--no-off-target", action="store_true",
                        help="Disable off-target scoring (ignore --reference-fasta)")
    parser.add_argument("--seed-len", type=int, default=None,
                        help="Override seed length (default: value stored in the checkpoint). Must keep "
                             "(20-seed_len)*4 equal to the trained action space.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max optimization steps per sequence (default: checkpoint value)")
    parser.add_argument("--max-mismatches", type=int, default=None,
                        help="Max mismatches from original (default: checkpoint value)")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"),
                        help="Model device (auto = cuda if available)")
    parser.add_argument("--max-sequences", type=int, default=0,
                        help="Maximum number of sequences to process (0 = all)")
    args = parser.parse_args()

    model_path = resolve_pt_path(args.model, _REPO_ROOT)
    if not os.path.isfile(model_path):
        print(f"Error: model not found: {model_path}")
        print("Hint: train_grna_rainbow.py saves to RL/models/grna_rainbow.pt")
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    out_path = args.output
    if not out_path:
        base, ext = os.path.splitext(args.input)
        out_path = base + "_optimized" + ext

    sequences, rows, fieldnames = load_sequences_from_csv(
        args.input, max_count=args.max_sequences if args.max_sequences else 100000
    )
    if not sequences:
        print("No valid 20-mer sequences found in the sgRNA column.")
        sys.exit(1)

    print(f"Loaded sequences: {len(sequences)}")
    print(f"Model: {model_path}")
    print(f"Output: {out_path}")

    import numpy as np
    import torch

    from RL.train_grna_rainbow import RainbowNet, build_env, env_reset, env_step, resolve_genome

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Device: {device}")

    # Load checkpoint and rebuild the network.
    # weights_only=False because the checkpoint stores a small config dict (incl. numpy
    # scalars) alongside the tensors. This is our own trusted checkpoint.
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    seed_len = args.seed_len if args.seed_len is not None else cfg["seed_len"]
    max_steps = args.max_steps if args.max_steps is not None else cfg["max_steps"]
    max_mismatches = args.max_mismatches if args.max_mismatches is not None else cfg["max_mismatches"]

    expected_actions = (20 - seed_len) * 4
    if expected_actions != cfg["n_actions"]:
        print(f"Error: seed_len={seed_len} implies {expected_actions} actions, but the model was trained "
              f"with {cfg['n_actions']} actions (seed_len={cfg['seed_len']}). Use --seed-len {cfg['seed_len']}.")
        sys.exit(1)

    net = RainbowNet(cfg["obs_dim"], cfg["n_actions"], cfg["n_atoms"], cfg["hidden"]).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()  # eval mode => NoisyLinear uses mean weights => deterministic policy
    support = torch.linspace(cfg["v_min"], cfg["v_max"], cfg["n_atoms"], device=device)
    print("Model loaded (Rainbow DQN).")

    # Off-target reference (loaded once)
    ref_fasta_arg = None if args.no_off_target else args.reference_fasta
    use_crisprspec = bool(ref_fasta_arg)
    genome_seq, ref_fasta = resolve_genome(use_crisprspec, ref_fasta_arg, None)
    if use_crisprspec and genome_seq is None and ref_fasta is None:
        # resolve_genome could not find the reference; try the data/ fallback
        fb = os.path.join(_REPO_ROOT, "data", os.path.basename(ref_fasta_arg or "genes_with_flankers.fna"))
        if os.path.isfile(fb):
            from RL.grna_rl_adapters import load_genome
            genome_seq = load_genome(fb)
            print(f"  [CRISPRspec] Genome loaded once: {len(genome_seq)} bp from {fb}")
        else:
            print("Off-target: disabled (reference FASTA not found)")
            use_crisprspec = False
    print(f"Off-target: {'enabled' if (use_crisprspec and genome_seq) else 'disabled'}")

    # Build a single env; reset per sequence with the exact input sequence
    env = build_env(
        sequences, seed_len, max_steps, max_mismatches, 1,
        use_crisprspec, genome_seq, 1.0, 0.1, 0.1,
        (str(device) == "cuda"), ref_fasta,
    )

    @torch.no_grad()
    def greedy_action(obs_np) -> int:
        state = torch.as_tensor(np.asarray(obs_np, dtype=np.float32).reshape(1, -1), device=device)
        dist = net(state)
        q = (dist * support).sum(dim=2)
        return int(q.argmax(dim=1).item())

    def reset_with(seq):
        out = env.reset(options={"sequence": seq})
        if isinstance(out, tuple):
            return out[0], out[1]
        return out, {}

    results = []
    for i, (seq, row) in enumerate(zip(sequences, rows)):
        obs, info = reset_with(seq)
        initial_score = info.get("score", None)
        step = 0
        last_info = info
        while step < max_steps:
            action = greedy_action(obs)
            obs, reward, terminated, truncated, info = env_step(env, action)
            last_info = info
            step += 1
            if terminated or truncated:
                break
        optimized = last_info.get("sequence", env.get_sequence())
        score = last_info.get("score", None)
        results.append({
            "row": row,
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
