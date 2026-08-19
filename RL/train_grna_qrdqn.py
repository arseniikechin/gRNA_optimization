"""
Train an RL agent (QR-DQN) for gRNA optimization.

Same environment / reward as the PPO script (CRISPRspec off-target + GC + homopolymer,
dense reward every step), but the policy is QR-DQN (Quantile Regression DQN) from
sb3-contrib instead of PPO.

QR-DQN is a value-based, off-policy algorithm for discrete action spaces, which matches
this environment's Discrete(mutable_len * 4) action space. It uses a replay buffer and
epsilon-greedy exploration (instead of PPO's on-policy rollouts + entropy bonus).

Install the extra dependency first:
    pip install sb3-contrib

Example (Linux/macOS), run from RL/:
    python train_grna_qrdqn.py \
      --sequences ../data/train_400.txt \
      --steps 200000 \
      --seed-len 8 \
      --max-episode-steps 20 \
      --use-crisprspec \
      --reference-fasta ../data/chr_3_7_20_21.fna
"""

import argparse
import os
import sys
from typing import List, Optional

# Add repo root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

# Core SB3 pieces (callbacks, vec env, monitor) live in stable-baselines3.
try:
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("Error: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3[extra]")
    sys.exit(1)

# QR-DQN lives in sb3-contrib.
try:
    from sb3_contrib import QRDQN
except ImportError:
    print("Error: sb3-contrib (QR-DQN) not installed.")
    print("Install with: pip install sb3-contrib")
    sys.exit(1)

try:
    import gymnasium as gym  # noqa: F401
    USING_GYMNASIUM = True
except ImportError:
    try:
        import gym  # noqa: F401
        USING_GYMNASIUM = False
    except ImportError:
        print("Error: Neither gymnasium nor gym installed.")
        print("Install with: pip install gymnasium  OR  pip install gym")
        sys.exit(1)

# Reuse the environment factory, sequence loaders and logging callback from the PPO
# script so the reward, env wiring and TensorBoard logging stay identical.
from RL.grna_gym_env import CRISPRGymEnv, BASES  # noqa: F401
from RL.train_grna_rl import (
    TrainingCallback,
    load_sequences,
    generate_random_sequences,
    make_env,
)

try:
    import torch
    _TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    _TORCH_CUDA_AVAILABLE = False


def train(
    sequences: List[str],
    total_timesteps: int = 50000,
    seed_len: int = 12,
    max_steps: int = 20,
    max_mismatches: int = 4,
    min_mismatches: int = 1,
    n_envs: int = 1,
    learning_rate: float = 1e-4,
    save_path: str = "RL/models/grna_qrdqn",
    log_dir: str = "RL/logs_qrdqn",
    use_crisprspec: bool = False,
    genome_path: Optional[str] = None,
    reference_fasta_path: Optional[str] = None,
    crisprspec_weight: float = 1.0,
    gc_weight: float = 0.1,
    homopolymer_weight: float = 0.1,
    eval_sequences: Optional[List[str]] = None,
    eval_freq: int = 5000,
    n_eval_episodes: int = 10,
    use_cuda: bool = True,
    # QR-DQN specific hyperparameters
    buffer_size: int = 100_000,
    learning_starts: int = 1000,
    batch_size: int = 64,
    tau: float = 1.0,
    gamma: float = 0.99,
    train_freq: int = 4,
    gradient_steps: int = 1,
    target_update_interval: int = 10_000,
    exploration_fraction: float = 0.2,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    n_quantiles: int = 200,
):
    """Train a QR-DQN agent on gRNA optimization (reward identical to the PPO script)."""
    device = "cuda" if (use_cuda and _TORCH_CUDA_AVAILABLE) else "cpu"
    use_cuda_offtarget = use_cuda

    print(f"Training gRNA RL agent (QR-DQN)")
    print(f"  Device: {device} (QR-DQN policy)")
    print(f"  Off-target search: {'CUDA (when available)' if use_cuda_offtarget else 'CPU'}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Parallel environments: {n_envs}")
    print()

    # Resolve the reference genome. Load it once here and share the same string object
    # across all environments instead of re-reading the FASTA in every env.
    genome_seq = None
    ref_fasta = reference_fasta_path
    if ref_fasta and not os.path.isabs(ref_fasta):
        ref_fasta = os.path.normpath(os.path.join(_REPO_ROOT, ref_fasta))
    if ref_fasta and not os.path.isfile(ref_fasta):
        ref_fasta = None
    if use_crisprspec and not ref_fasta:
        _genome_path = genome_path
        if genome_path and not os.path.isfile(genome_path) and not os.path.isabs(genome_path):
            _alt = os.path.join(_REPO_ROOT, genome_path)
            if os.path.isfile(_alt):
                _genome_path = _alt
        if _genome_path and os.path.isfile(_genome_path):
            from RL.grna_rl_adapters import load_genome
            genome_seq = load_genome(_genome_path)
            print(f"  [CRISPRspec] Genome loaded: {len(genome_seq)} bp")
        else:
            print(f"  WARNING: --use-crisprspec specified but genome not loaded!")
            if not genome_path:
                print(f"           Provide --genome <path> or --reference-fasta <path>")
            else:
                print(f"           Genome file not found: {genome_path}")
            print(f"           Training will continue but CRISPRspec scores will be 0")
    elif use_crisprspec and ref_fasta:
        from RL.grna_rl_adapters import load_genome
        genome_seq = load_genome(ref_fasta)
        print(f"  [CRISPRspec] Genome loaded once: {len(genome_seq)} bp from {ref_fasta}")
        ref_fasta = None  # pass genome_seq directly; avoid per-env reload

    print(f"  Mutable region: positions 0-{20-seed_len-1} ({20-seed_len} positions)")
    print(f"  Seed region (fixed): positions {20-seed_len}-19 ({seed_len} positions)")
    print(f"  Mismatches allowed: {min_mismatches}-{max_mismatches}")
    print(f"  Reward components (only CRISPRspec + GC + homopolymer):")
    print(f"    - CRISPRspec: {'ENABLED' if (use_crisprspec and (genome_seq or ref_fasta)) else 'DISABLED'} (weight={crisprspec_weight})")
    print(f"    - GC content: ENABLED (weight={gc_weight})")
    print(f"    - Homopolymers: ENABLED (weight={homopolymer_weight})")
    print(f"  QR-DQN exploration: eps {exploration_initial_eps} -> {exploration_final_eps} over {exploration_fraction*100:.0f}% of training")
    print(f"  QR-DQN quantiles: {n_quantiles}")

    # Build (vectorized) training environment. make_env already wires the CLI weights
    # into the actual reward terms.
    env = DummyVecEnv([
        make_env(
            sequences, seed_len, max_steps, max_mismatches, min_mismatches,
            use_crisprspec, genome_seq, crisprspec_weight, gc_weight, homopolymer_weight,
            use_cuda_offtarget, True, i, ref_fasta,
        )
        for i in range(max(1, n_envs))
    ])

    # Create QR-DQN agent
    print("Creating QR-DQN agent...")
    model = QRDQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs={"n_quantiles": n_quantiles},
        verbose=0,  # our own callback handles progress printing
        tensorboard_log=log_dir,
        device=device,
    )
    print(f"  Policy: MlpPolicy / QR-DQN (device={device})")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Buffer: {buffer_size}, batch: {batch_size}, train_freq: {train_freq}, target_update: {target_update_interval}")
    print()

    # Logging callback (same as PPO script); saves training_summary.{json,txt} to log_dir.
    callback = TrainingCallback(verbose=1, log_freq=1000, target_timesteps=total_timesteps, log_dir=log_dir)
    callback.model = model

    callbacks = [callback]
    eval_env = None
    if eval_sequences:
        print(f"  [Evaluation] Validation set: {len(eval_sequences)} sequences")
        print(f"  [Evaluation] Eval frequency: every {eval_freq} steps, {n_eval_episodes} episodes")
        eval_env = DummyVecEnv([
            make_env(
                eval_sequences, seed_len, max_steps, max_mismatches, min_mismatches,
                use_crisprspec, genome_seq, crisprspec_weight, gc_weight, homopolymer_weight,
                use_cuda_offtarget, True, 0, ref_fasta,
            )
        ])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path + "_best",
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )
        callbacks.append(eval_callback)
        print(f"  [Evaluation] Best model will be saved to: {save_path}_best.zip")
    else:
        print(f"  [Evaluation] No validation set provided (use --eval-sequences for validation)")

    print("Starting training...")
    print(f"  Target timesteps: {total_timesteps:,}")
    print("-" * 50)
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[Training interrupted by user]")
        raise
    except Exception as e:
        print(f"\n[Training error: {e}]")
        import traceback
        traceback.print_exc()
        raise
    print("-" * 50)
    print("Training complete!")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    if eval_sequences:
        print(f"Best model (based on validation) saved to: {save_path}_best.zip")

    env.close()
    if eval_env is not None:
        eval_env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train QR-DQN agent for gRNA optimization.")
    parser.add_argument("--sequences", "-s", type=str, default=None,
                        help="Path to file with training sequences (one 20-mer per line, or CSV with sgRNA column).")
    parser.add_argument("--random-seqs", type=int, default=0,
                        help="Generate N random sequences for training (if --sequences not provided).")
    parser.add_argument("--steps", type=int, default=50000,
                        help="Total training timesteps (default: 50000).")
    parser.add_argument("--seed-len", type=int, default=8,
                        help="Fix LAST N positions (seed region near PAM). Default: 8.")
    parser.add_argument("--max-mismatches", type=int, default=4,
                        help="Maximum mismatches from original sequence (default: 4).")
    parser.add_argument("--min-mismatches", type=int, default=1,
                        help="Minimum mismatches required (default: 1).")
    parser.add_argument("--max-episode-steps", type=int, default=20,
                        help="Max steps per episode (default: 20).")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments (default: 1; QR-DQN is off-policy).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4).")
    parser.add_argument("--save-path", type=str,
                        default=os.path.join(_SCRIPT_DIR, "models", "grna_qrdqn"),
                        help="Path to save trained model.")
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join(_SCRIPT_DIR, "logs_qrdqn"),
                        help="Directory for tensorboard logs.")
    parser.add_argument("--use-crisprspec", action="store_true",
                        help="Include CRISPRspec surrogate in reward for off-target specificity.")
    parser.add_argument("--genome", type=str, default=None,
                        help="Genome FASTA for off-target search (alternative to --reference-fasta).")
    parser.add_argument("--reference-fasta", type=str, default=None,
                        help="Path to reference FASTA for CRISPRspec; overrides --genome.")
    parser.add_argument("--crisprspec-weight", type=float, default=1.0,
                        help="Weight for CRISPRspec in reward (default: 1).")
    parser.add_argument("--gc-weight", type=float, default=0.1,
                        help="Weight for GC penalty in reward (default: 0.1).")
    parser.add_argument("--homopolymer-weight", type=float, default=0.1,
                        help="Weight for homopolymer penalty in reward (default: 0.1).")
    parser.add_argument("--eval-sequences", type=str, default=None,
                        help="Path to validation sequences (one 20-mer per line).")
    parser.add_argument("--eval-freq", type=int, default=5000,
                        help="Evaluate every N steps (default: 5000).")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                        help="Episodes per evaluation (default: 10).")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Use CPU for the policy and off-target search.")
    # QR-DQN specific
    parser.add_argument("--buffer-size", type=int, default=100_000,
                        help="Replay buffer size (default: 100000).")
    parser.add_argument("--learning-starts", type=int, default=1000,
                        help="Steps collected before learning starts (default: 1000).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Minibatch size (default: 64).")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Target network soft-update coefficient (default: 1.0 = hard update).")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99).")
    parser.add_argument("--train-freq", type=int, default=4,
                        help="Gradient update every N steps (default: 4).")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="Gradient steps per update (default: 1).")
    parser.add_argument("--target-update-interval", type=int, default=10_000,
                        help="Steps between target network updates (default: 10000).")
    parser.add_argument("--exploration-fraction", type=float, default=0.2,
                        help="Fraction of training over which epsilon is annealed (default: 0.2).")
    parser.add_argument("--exploration-final-eps", type=float, default=0.05,
                        help="Final epsilon for epsilon-greedy exploration (default: 0.05).")
    parser.add_argument("--n-quantiles", type=int, default=200,
                        help="Number of quantiles in QR-DQN (default: 200).")
    args = parser.parse_args()

    # Load or generate sequences
    if args.sequences:
        if not os.path.isfile(args.sequences):
            print(f"Error: File not found: {args.sequences}")
            sys.exit(1)
        sequences = load_sequences(args.sequences)
        print(f"Loaded {len(sequences)} sequences from {args.sequences}")
    elif args.random_seqs > 0:
        sequences = generate_random_sequences(args.random_seqs)
        print(f"Generated {len(sequences)} random sequences")
    else:
        sequences = generate_random_sequences(50)
        print(f"Generated 50 random sequences (use --sequences or --random-seqs to customize)")

    if not sequences:
        print("Error: No valid sequences found.")
        sys.exit(1)

    eval_sequences = None
    if args.eval_sequences:
        if not os.path.isfile(args.eval_sequences):
            print(f"Warning: Evaluation sequences file not found: {args.eval_sequences}")
            print("         Training will continue without validation.")
        else:
            eval_sequences = load_sequences(args.eval_sequences)
            print(f"Loaded {len(eval_sequences)} validation sequences from {args.eval_sequences}")

    train(
        sequences=sequences,
        total_timesteps=args.steps,
        seed_len=args.seed_len,
        max_steps=args.max_episode_steps,
        max_mismatches=args.max_mismatches,
        min_mismatches=args.min_mismatches,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        save_path=args.save_path,
        log_dir=args.log_dir,
        use_crisprspec=args.use_crisprspec,
        genome_path=args.genome,
        reference_fasta_path=args.reference_fasta,
        crisprspec_weight=args.crisprspec_weight,
        gc_weight=args.gc_weight,
        homopolymer_weight=args.homopolymer_weight,
        eval_sequences=eval_sequences,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        use_cuda=not args.force_cpu,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        n_quantiles=args.n_quantiles,
    )


if __name__ == "__main__":
    main()
