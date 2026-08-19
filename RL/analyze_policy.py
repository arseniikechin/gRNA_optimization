#!/usr/bin/env python3
"""
Analyze a trained gRNA RL policy and export statistics.

Usage:
    python analyze_policy.py --model models/grna_ppo.zip --sequences eval_100.txt \
        --genome ../data/chr_3_7_20_21.fna --output analysis_results/

Generates:
    - policy_summary.json: all statistics in machine-readable format
    - policy_report.txt: human-readable summary
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

BASES = ["A", "C", "G", "T"]
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


class _RainbowPredictor:
    """Thin wrapper around a trained Rainbow network exposing an SB3-like predict()."""

    def __init__(self, ckpt_path: str, device):
        import torch
        from RL.train_grna_rainbow import RainbowNet
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.net = RainbowNet(cfg["obs_dim"], cfg["n_actions"], cfg["n_atoms"], cfg["hidden"]).to(device)
        self.net.load_state_dict(ckpt["model_state"])
        self.net.eval()  # NoisyLinear uses mean weights => deterministic policy
        self.support = torch.linspace(cfg["v_min"], cfg["v_max"], cfg["n_atoms"], device=device)
        self.device = device
        self.config = cfg

    def predict(self, obs, deterministic: bool = True):
        import torch
        with torch.no_grad():
            state = torch.as_tensor(
                np.asarray(obs, dtype=np.float32).reshape(1, -1), device=self.device
            )
            dist = self.net(state)
            q = (dist * self.support).sum(dim=2)
            action = int(q.argmax(dim=1).item())
        return action, None


def load_agent(model_path: str, device: str = "auto"):
    """
    Load a trained agent, auto-detecting the algorithm.

    Returns (model, kind, config) where:
      - kind is one of "ppo", "qrdqn", "rainbow"
      - config is the Rainbow checkpoint config dict (else None)
    All returned models expose .predict(obs, deterministic=...) -> (action, state).
    """
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Resolve a bare path (no extension) to the actual checkpoint on disk.
    if not os.path.isfile(model_path):
        for ext in (".pt", ".zip"):
            if os.path.isfile(model_path + ext):
                model_path = model_path + ext
                break

    # Detect a Rainbow checkpoint by *content*, not extension: torch.save writes
    # a zip whose entries end in "data.pkl" and which lacks SB3 marker files.
    # (A trained Rainbow model may be stored as either .pt or .zip.)
    is_rainbow = model_path.endswith(".pt")
    if not is_rainbow:
        try:
            import zipfile
            if zipfile.is_zipfile(model_path):
                names = zipfile.ZipFile(model_path).namelist()
                is_sb3 = any("_stable_baselines3_version" in n for n in names) or "data" in names
                has_torch_pickle = any(n.endswith("data.pkl") for n in names)
                is_rainbow = has_torch_pickle and not is_sb3
        except Exception:
            is_rainbow = False

    if is_rainbow:
        predictor = _RainbowPredictor(model_path, device)
        return predictor, "rainbow", predictor.config

    # Otherwise a Stable-Baselines3 .zip: try PPO, then QR-DQN
    last_err = None
    try:
        from stable_baselines3 import PPO
        return PPO.load(model_path, device=device), "ppo", None
    except Exception as e:
        last_err = e
    try:
        from sb3_contrib import QRDQN
        return QRDQN.load(model_path, device=device), "qrdqn", None
    except Exception as e:
        raise RuntimeError(
            f"Could not load model as PPO or QR-DQN: {e}. "
            f"(PPO attempt failed with: {last_err})"
        )


def load_sequences(path: str) -> List[str]:
    """Load 20-mer gRNA sequences from file."""
    seqs = []
    with open(path) as f:
        for line in f:
            s = line.strip().upper().replace("U", "T")
            if len(s) == 20 and all(c in "ACGT" for c in s):
                seqs.append(s)
    return seqs


def run_policy_on_sequences(
    model,
    sequences: List[str],
    seed_len: int = 8,
    max_steps: int = 20,
    max_mismatches: int = 4,
    genome_seq: Optional[str] = None,
    reference_fasta_path: Optional[str] = None,
    use_crisprspec: bool = True,
    deterministic: bool = True,
) -> List[Dict]:
    """
    Run trained policy on a list of sequences, record every action.
    Returns list of episode records.
    """
    from RL.grna_gym_env import CRISPRGymEnv, seq_to_onehot

    env = CRISPRGymEnv(
        initial_sequences=sequences,
        seed_len=seed_len,
        max_steps=max_steps,
        max_mismatches=max_mismatches,
        use_crisprspec=use_crisprspec,
        genome_seq=genome_seq,
        reference_fasta_path=reference_fasta_path,
        use_cuda_offtarget=True,
    )

    episodes = []
    for seq_idx, seq in enumerate(sequences):
        obs, info = env.reset(options={"sequence": seq})
        original_seq = seq
        episode = {
            "original_sequence": original_seq,
            "actions": [],
            "mutations": [],  # (step, pos, old_base, new_base)
            "sequences": [original_seq],
            "scores": [info.get("score", 0.0)],
            "components": [info.get("components", {}).copy()],
        }

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            pos = action // 4
            base_idx = action % 4
            new_base = BASES[base_idx]
            old_base = env.sequence[pos] if pos < len(env.sequence) else "?"

            obs, reward, terminated, truncated, info = env.step(action)
            current_seq = env.sequence

            episode["actions"].append({
                "step": step,
                "action": action,
                "position": pos,
                "new_base": new_base,
                "old_base": old_base,
                "is_mutation": old_base != new_base,
                "is_noop": old_base == new_base,
                "reward": float(reward),
            })

            if old_base != new_base and current_seq != episode["sequences"][-1]:
                episode["mutations"].append((step, pos, old_base, new_base))

            episode["sequences"].append(current_seq)
            episode["scores"].append(info.get("score", 0.0))
            episode["components"].append(info.get("components", {}).copy())

            if terminated or truncated:
                break

        episode["final_sequence"] = env.sequence
        episode["initial_score"] = episode["scores"][0]
        episode["final_score"] = episode["scores"][-1]
        episode["score_improvement"] = episode["final_score"] - episode["initial_score"]
        episode["n_mutations"] = len(episode["mutations"])
        episode["mismatches"] = sum(
            1 for a, b in zip(original_seq, env.sequence) if a != b
        )
        episodes.append(episode)

        if (seq_idx + 1) % 50 == 0:
            print(f"  Processed {seq_idx + 1}/{len(sequences)} sequences...")

    env.close()
    return episodes


def analyze_episodes(episodes: List[Dict], mutable_len: int = 12) -> Dict:
    """Compute aggregate statistics from episode records."""
    stats = {}

    # --- 1. Position mutation frequency ---
    pos_counts = np.zeros(mutable_len, dtype=int)
    pos_base_counts = np.zeros((mutable_len, 4), dtype=int)  # pos × target_base
    transition_counts = np.zeros((4, 4), dtype=int)  # old_base × new_base
    # NEW: position-resolved from→to tensor (mutable_len × 4 × 4)
    pos_from_to_counts = np.zeros((mutable_len, 4, 4), dtype=int)
    all_mutations = []

    for ep in episodes:
        for step, pos, old_base, new_base in ep["mutations"]:
            if pos < mutable_len:
                pos_counts[pos] += 1
                pos_base_counts[pos, BASE_TO_IDX[new_base]] += 1
                transition_counts[BASE_TO_IDX[old_base], BASE_TO_IDX[new_base]] += 1
                pos_from_to_counts[pos, BASE_TO_IDX[old_base], BASE_TO_IDX[new_base]] += 1
                all_mutations.append({
                    "pos": pos, "old": old_base, "new": new_base,
                    "context_before": ep["original_sequence"][max(0, pos-2):pos+3],
                })

    total_mutations = len(all_mutations)
    stats["total_mutations"] = total_mutations
    stats["total_episodes"] = len(episodes)
    stats["avg_mutations_per_episode"] = total_mutations / max(1, len(episodes))

    # Position frequency (normalized)
    stats["position_frequency"] = (pos_counts / max(1, total_mutations)).tolist()
    stats["position_counts"] = pos_counts.tolist()

    # Position × base heatmap (normalized per position)
    pos_base_norm = np.zeros_like(pos_base_counts, dtype=float)
    for i in range(mutable_len):
        s = pos_base_counts[i].sum()
        if s > 0:
            pos_base_norm[i] = pos_base_counts[i] / s
    stats["position_base_preference"] = pos_base_norm.tolist()
    stats["position_base_counts"] = pos_base_counts.tolist()

    # Transition matrix (normalized per old_base)
    trans_norm = np.zeros_like(transition_counts, dtype=float)
    for i in range(4):
        s = transition_counts[i].sum()
        if s > 0:
            trans_norm[i] = transition_counts[i] / s
    stats["transition_matrix"] = trans_norm.tolist()
    stats["transition_counts"] = transition_counts.tolist()

    # NEW: position-resolved from→to counts  (shape: mutable_len × 4 × 4)
    # Each entry [pos][from_base][to_base] = number of accepted mutations
    stats["position_from_to_counts"] = pos_from_to_counts.tolist()

    # NEW: flat mutation log — every accepted mutation with full context
    stats["mutation_log"] = [
        {"pos": m["pos"], "old": m["old"], "new": m["new"]}
        for m in all_mutations
    ]

    # --- 2. No-op analysis ---
    noop_count = sum(
        1 for ep in episodes for a in ep["actions"] if a["is_noop"]
    )
    total_actions = sum(len(ep["actions"]) for ep in episodes)
    stats["noop_fraction"] = noop_count / max(1, total_actions)
    stats["total_actions"] = total_actions

    # --- 3. Score improvements ---
    improvements = [ep["score_improvement"] for ep in episodes]
    stats["score_improvement_mean"] = float(np.mean(improvements))
    stats["score_improvement_std"] = float(np.std(improvements))
    stats["score_improvement_median"] = float(np.median(improvements))
    stats["pct_improved"] = float(np.mean([x > 0 for x in improvements]))

    # --- 4. Component analysis ---
    initial_crisprspec = []
    final_crisprspec = []
    initial_gc = []
    final_gc = []
    initial_hp = []
    final_hp = []
    initial_offtargets = []
    final_offtargets = []

    for ep in episodes:
        if ep["components"]:
            c0 = ep["components"][0]
            cf = ep["components"][-1]
            initial_crisprspec.append(c0.get("crisprspec", 0))
            final_crisprspec.append(cf.get("crisprspec", 0))
            initial_gc.append(c0.get("gc_content", 0))
            final_gc.append(cf.get("gc_content", 0))
            initial_hp.append(c0.get("homopolymer_term", 0))
            final_hp.append(cf.get("homopolymer_term", 0))
            initial_offtargets.append(c0.get("n_off_targets", 0))
            final_offtargets.append(cf.get("n_off_targets", 0))

    stats["crisprspec"] = {
        "initial_mean": float(np.mean(initial_crisprspec)) if initial_crisprspec else 0,
        "final_mean": float(np.mean(final_crisprspec)) if final_crisprspec else 0,
        "improvement": float(np.mean(final_crisprspec) - np.mean(initial_crisprspec)) if initial_crisprspec else 0,
    }
    stats["gc_content"] = {
        "initial_mean": float(np.mean(initial_gc)) if initial_gc else 0,
        "final_mean": float(np.mean(final_gc)) if final_gc else 0,
    }
    stats["off_targets"] = {
        "initial_mean": float(np.mean(initial_offtargets)) if initial_offtargets else 0,
        "final_mean": float(np.mean(final_offtargets)) if final_offtargets else 0,
    }

    # --- 5. Most common mutations ---
    mutation_strs = [f"{m['old']}>{m['new']} @pos{m['pos']}" for m in all_mutations]
    stats["top_mutations"] = Counter(mutation_strs).most_common(20)

    # Substitution type counts (e.g. A>C, G>T)
    sub_types = [f"{m['old']}>{m['new']}" for m in all_mutations]
    stats["substitution_types"] = dict(Counter(sub_types).most_common())

    # --- 6. Per-sequence data for downstream statistical analysis ---
    stats["per_sequence"] = {
        "initial_scores": [ep["initial_score"] for ep in episodes],
        "final_scores": [ep["final_score"] for ep in episodes],
        "improvements": improvements,
        "n_mutations": [ep["n_mutations"] for ep in episodes],
        "initial_crisprspec": initial_crisprspec,
        "final_crisprspec": final_crisprspec,
        "initial_gc": initial_gc,
        "final_gc": final_gc,
        "initial_offtargets": initial_offtargets,
        "final_offtargets": final_offtargets,
    }

    return stats


def generate_report(stats: Dict, output_dir: str):
    """Generate human-readable report."""
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("gRNA RL POLICY ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total episodes analyzed: {stats['total_episodes']}")
    lines.append(f"Total mutations applied: {stats['total_mutations']}")
    lines.append(f"Average mutations per episode: {stats['avg_mutations_per_episode']:.2f}")
    lines.append(f"No-op fraction: {stats['noop_fraction']:.3f} ({stats['noop_fraction']*100:.1f}%)")
    lines.append("")

    lines.append("--- Score Improvement ---")
    lines.append(f"Mean improvement: {stats['score_improvement_mean']:.4f}")
    lines.append(f"Std: {stats['score_improvement_std']:.4f}")
    lines.append(f"Median: {stats['score_improvement_median']:.4f}")
    lines.append(f"Sequences improved: {stats['pct_improved']*100:.1f}%")
    lines.append("")

    lines.append("--- CRISPRspec ---")
    cs = stats["crisprspec"]
    lines.append(f"Initial mean: {cs['initial_mean']:.4f}")
    lines.append(f"Final mean:   {cs['final_mean']:.4f}")
    lines.append(f"Improvement:  {cs['improvement']:.4f}")
    lines.append("")

    lines.append("--- GC Content ---")
    gc = stats["gc_content"]
    lines.append(f"Initial mean: {gc['initial_mean']:.3f}")
    lines.append(f"Final mean:   {gc['final_mean']:.3f}")
    lines.append("")

    lines.append("--- Off-targets ---")
    ot = stats["off_targets"]
    lines.append(f"Initial mean: {ot['initial_mean']:.1f}")
    lines.append(f"Final mean:   {ot['final_mean']:.1f}")
    lines.append("")

    lines.append("--- Top 20 Most Common Mutations ---")
    for mut, count in stats["top_mutations"]:
        lines.append(f"  {mut}: {count}")
    lines.append("")

    lines.append("--- Substitution Type Distribution ---")
    for sub, count in sorted(stats["substitution_types"].items(), key=lambda x: -x[1]):
        lines.append(f"  {sub}: {count}")
    lines.append("")

    lines.append("--- Position Mutation Counts ---")
    for i, c in enumerate(stats["position_counts"]):
        bar = "█" * (c // 5) if c > 0 else ""
        lines.append(f"  pos {i:2d}: {c:4d} {bar}")
    lines.append("")

    report = "\n".join(lines)

    report_path = os.path.join(output_dir, "policy_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")
    print()
    print(report)

    # Also save JSON
    json_path = os.path.join(output_dir, "policy_summary.json")
    # Remove per_sequence for smaller JSON (it's in the full stats)
    json_stats = {k: v for k, v in stats.items() if k != "per_sequence"}
    with open(json_path, "w") as f:
        json.dump(json_stats, f, indent=2, default=str)
    print(f"  JSON saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze trained gRNA RL policy")
    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--sequences", required=True, help="Path to evaluation sequences file")
    parser.add_argument("--genome", default=None, help="Path to genome FASTA for CRISPRspec")
    parser.add_argument("--output", default="analysis_results", help="Output directory")
    parser.add_argument("--seed-len", type=int, default=8, help="Seed length (fixed positions)")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--max-mismatches", type=int, default=4, help="Max mismatches")
    parser.add_argument("--no-crisprspec", action="store_true", help="Disable CRISPRspec")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy (default: True)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (sample from distribution)")
    parser.add_argument("--device", default="auto", help="Device for inference (auto/cpu/cuda)")
    args = parser.parse_args()

    deterministic = not args.stochastic

    print("=" * 50)
    print("gRNA RL Policy Analysis")
    print("=" * 50)

    # Load model (auto-detect PPO / QR-DQN / Rainbow)
    print(f"\nLoading model from {args.model}...")
    model, kind, cfg = load_agent(args.model, device=args.device)
    print(f"  Model loaded successfully (type: {kind.upper()})")

    # Align the analysis env to the model to avoid action-space / seed mismatches.
    # Rainbow checkpoints carry the full env config; SB3 models encode the
    # mutable region in their action space (Discrete((20 - seed_len) * 4)).
    if kind == "rainbow" and cfg is not None:
        for key, arg_name in (("seed_len", "seed_len"),
                              ("max_steps", "max_steps"),
                              ("max_mismatches", "max_mismatches")):
            if key in cfg and getattr(args, arg_name) != cfg[key]:
                print(f"  [{kind}] overriding --{arg_name.replace('_', '-')} "
                      f"{getattr(args, arg_name)} -> {cfg[key]} (from checkpoint)")
                setattr(args, arg_name, cfg[key])
    else:
        try:
            inferred_seed_len = 20 - (int(model.action_space.n) // 4)
            if inferred_seed_len != args.seed_len:
                print(f"  [{kind}] overriding --seed-len {args.seed_len} -> "
                      f"{inferred_seed_len} (inferred from action space)")
                args.seed_len = inferred_seed_len
        except Exception:
            pass

    # Load sequences
    sequences = load_sequences(args.sequences)
    print(f"  Loaded {len(sequences)} sequences from {args.sequences}")

    # Load genome
    genome_seq = None
    reference_fasta_path = None
    if args.genome and os.path.isfile(args.genome):
        reference_fasta_path = args.genome
        from RL.grna_rl_adapters import load_genome
        genome_seq = load_genome(args.genome)
        print(f"  Genome loaded: {len(genome_seq)} bp from {args.genome}")

    mutable_len = 20 - args.seed_len

    # Run policy
    print(f"\nRunning policy on {len(sequences)} sequences...")
    print(f"  Mutable positions: 0-{mutable_len - 1} ({mutable_len} positions)")
    print(f"  Max steps: {args.max_steps}, Max mismatches: {args.max_mismatches}")
    print(f"  Mode: {'deterministic' if deterministic else 'stochastic'}")

    episodes = run_policy_on_sequences(
        model=model,
        sequences=sequences,
        seed_len=args.seed_len,
        max_steps=args.max_steps,
        max_mismatches=args.max_mismatches,
        genome_seq=genome_seq,
        reference_fasta_path=reference_fasta_path,
        use_crisprspec=not args.no_crisprspec and genome_seq is not None,
        deterministic=deterministic,
    )

    # Analyze
    print(f"\nAnalyzing {len(episodes)} episodes...")
    stats = analyze_episodes(episodes, mutable_len=mutable_len)

    # Generate outputs (statistics only; no images)
    print(f"\nGenerating report...")
    generate_report(stats, args.output)

    print(f"\nDone! Results in {args.output}/")


if __name__ == "__main__":
    main()
