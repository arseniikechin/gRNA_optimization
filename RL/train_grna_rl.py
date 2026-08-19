"""
Train an RL agent (PPO) for gRNA optimization.

Reward: only CRISPRspec (off-target) + GC content + homopolymer.
Dense reward by default (reward every step = score change).

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

# Check for required packages
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("Error: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3[extra]")
    sys.exit(1)

try:
    import gymnasium as gym
    USING_GYMNASIUM = True
except ImportError:
    try:
        import gym
        USING_GYMNASIUM = False
    except ImportError:
        print("Error: Neither gymnasium nor gym installed.")
        print("Install with: pip install gymnasium  OR  pip install gym")
        sys.exit(1)

from RL.grna_gym_env import CRISPRGymEnv, make_training_env, BASES

try:
    import torch
    _TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    _TORCH_CUDA_AVAILABLE = False


class TrainingCallback(BaseCallback):
    """Custom callback to log training progress."""
    
    def __init__(self, verbose=1, log_freq=500, target_timesteps=None, log_dir=None):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.target_timesteps = target_timesteps
        self.log_dir = log_dir  # Where to save training_summary.json / .txt
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_count = 0
        self.start_time = None
        self.start_datetime_iso = None
        self.model = None  # Will be set by SB3
        # Component tracking
        self.episode_crisprspec = []
        self.episode_gc_content = []
        self.episode_homopolymer = []
        self.episode_n_offtargets = []
    
    def _on_training_start(self) -> None:
        import time
        from datetime import datetime
        self.start_time = time.time()
        self.start_datetime_iso = datetime.now().isoformat()
        self.last_print_time = time.time()
        # Store reference to model for later access
        if hasattr(self, 'model') and self.model is None:
            self.model = self.locals.get('self') if self.locals else None
        print(f"[Training started] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Computing first steps (this may take a while)...")
        import sys
        sys.stdout.flush()
    
    def _on_step(self) -> bool:
        import time
        import sys
        
        # Log episode info when available
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_count += 1
                if "score" in info:
                    self.episode_scores.append(info["score"])
                # Log score components
                if "components" in info:
                    components = info["components"]
                    self.episode_crisprspec.append(components.get("crisprspec", 0.0))
                    self.episode_gc_content.append(components.get("gc_content", 0.0))
                    self.episode_homopolymer.append(components.get("homopolymer_term", 0.0))
                    self.episode_n_offtargets.append(components.get("n_off_targets", 0))
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        time_since_print = time.time() - self.last_print_time
        
        # Print every log_freq steps OR every 10 seconds OR first 10 steps
        should_print = (
            self.n_calls % self.log_freq == 0 or 
            time_since_print > 10 or 
            self.n_calls <= 10
        )
        
        if should_print and self.verbose > 0:
            self.last_print_time = time.time()
            steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0
            
            # Simple progress indicator
            if self.n_calls <= 10:
                print(f"  Step {self.n_calls}... ({elapsed:.1f}s)")
            else:
                # Get actual timesteps from SB3 (more accurate than estimate)
                # Try multiple ways to get the real timestep count
                actual_timesteps = None
                if self.model and hasattr(self.model, 'num_timesteps'):
                    actual_timesteps = self.model.num_timesteps
                elif self.locals:
                    actual_timesteps = self.locals.get("num_timesteps")
                
                if actual_timesteps is None:
                    # Fallback: estimate based on n_calls and typical n_steps
                    actual_timesteps = self.n_calls * 1024
                
                progress = ""
                if self.target_timesteps:
                    pct = min(100, (actual_timesteps / self.target_timesteps) * 100)
                    progress = f" | {actual_timesteps:,}/{self.target_timesteps:,} ({pct:.1f}%)"
                print(f"\n[Update {self.n_calls:,}] "
                      f"Timesteps: {actual_timesteps:,}{progress} | "
                      f"Eps: {self.episode_count} | "
                      f"{steps_per_sec:.2f} updates/s | "
                      f"{elapsed:.0f}s elapsed")
                
                if self.episode_rewards:
                    recent_rewards = self.episode_rewards[-50:]
                    mean_reward = np.mean(recent_rewards)
                    print(f"  Reward: {mean_reward:.4f} (last {len(recent_rewards)} eps)")
                    # Log to TensorBoard
                    self.logger.record("train/ep_reward_mean", mean_reward)
                
                if self.episode_scores:
                    recent_scores = self.episode_scores[-50:]
                    mean_score = np.mean(recent_scores)
                    print(f"  Score: {mean_score:.4f}")
                    # Log to TensorBoard
                    self.logger.record("train/ep_score_mean", mean_score)
                
                # Log score components
                if self.episode_crisprspec:
                    recent_cs = self.episode_crisprspec[-50:]
                    recent_gc = self.episode_gc_content[-50:] if len(self.episode_gc_content) >= 50 else self.episode_gc_content
                    recent_hp = self.episode_homopolymer[-50:] if len(self.episode_homopolymer) >= 50 else self.episode_homopolymer
                    recent_ot = self.episode_n_offtargets[-50:] if len(self.episode_n_offtargets) >= 50 else self.episode_n_offtargets
                    print(f"  Components (last {len(recent_cs)} eps):")
                    print(f"    CRISPRspec: {np.mean(recent_cs):.4f} | "
                          f"GC: {np.mean(recent_gc):.3f} | "
                          f"Homopolymer: {np.mean(recent_hp):.3f} | "
                          f"Off-targets: {np.mean(recent_ot):.1f}")
                    # Log components to TensorBoard
                    self.logger.record("train/crisprspec_mean", np.mean(recent_cs))
                    self.logger.record("train/gc_content_mean", np.mean(recent_gc))
                    self.logger.record("train/homopolymer_mean", np.mean(recent_hp))
                    self.logger.record("train/offtargets_mean", np.mean(recent_ot))
                
                # Dump logs to TensorBoard
                self.logger.record("train/episodes", self.episode_count)
                self.logger.record("train/timesteps", actual_timesteps)
                self.logger.dump(step=actual_timesteps)
            
            sys.stdout.flush()
        
        return True
    
    def _on_training_end(self) -> None:
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0
        # Get final timesteps count from SB3
        # Try to get from model if available, otherwise estimate
        final_timesteps = None
        if hasattr(self, 'model') and hasattr(self.model, 'num_timesteps'):
            final_timesteps = self.model.num_timesteps
        elif hasattr(self, 'locals') and self.locals:
            final_timesteps = self.locals.get('num_timesteps')
        
        if final_timesteps is None:
            # Fallback estimate
            final_timesteps = self.n_calls * 1024
        
        # Human-readable duration
        total_min, total_sec = divmod(elapsed, 60)
        total_hr, total_min = divmod(total_min, 60)
        if total_hr >= 1:
            time_str = f"{int(total_hr)}h {int(total_min)}m {total_sec:.0f}s"
        elif total_min >= 1:
            time_str = f"{int(total_min)}m {total_sec:.0f}s"
        else:
            time_str = f"{elapsed:.1f}s"
        print(f"\n[Training finished]")
        print(f"  Total policy updates (n_calls): {self.n_calls:,}")
        print(f"  Total timesteps: {final_timesteps:,}")
        if self.target_timesteps:
            pct = (final_timesteps / self.target_timesteps) * 100
            status = "COMPLETED" if pct >= 99.5 else "STOPPED EARLY"
            print(f"  Target was: {self.target_timesteps:,} ({pct:.1f}% completed) [{status}]")
        print(f"  Total episodes: {self.episode_count}")
        print(f"  Total time: {elapsed:.1f}s ({time_str})")
        if self.episode_rewards:
            print(f"  Final mean reward: {np.mean(self.episode_rewards[-100:]):.4f}")
        if self.episode_scores:
            print(f"  Final mean score: {np.mean(self.episode_scores[-100:]):.4f}")
        # Final component statistics
        if self.episode_crisprspec:
            print(f"  Final components (last 100 eps):")
            print(f"    CRISPRspec: {np.mean(self.episode_crisprspec[-100:]):.4f}")
            print(f"    GC content: {np.mean(self.episode_gc_content[-100:]):.3f}")
            print(f"    Homopolymer term: {np.mean(self.episode_homopolymer[-100:]):.3f}")
            print(f"    Avg off-targets: {np.mean(self.episode_n_offtargets[-100:]):.1f}")

        # Save summary to log_dir (training_summary.json + training_summary.txt)
        if self.log_dir:
            from datetime import datetime
            import json
            end_iso = datetime.now().isoformat()
            summary = {
                "training_start": self.start_datetime_iso,
                "training_end": end_iso,
                "total_time_seconds": round(elapsed, 2),
                "total_time_human": time_str,
                "total_timesteps": int(final_timesteps),
                "target_timesteps": int(self.target_timesteps) if self.target_timesteps else None,
                "completion_percent": round((final_timesteps / self.target_timesteps) * 100, 1) if self.target_timesteps else None,
                "status": "COMPLETED" if (self.target_timesteps and final_timesteps >= 0.995 * self.target_timesteps) else "STOPPED_EARLY",
                "total_episodes": self.episode_count,
                "policy_updates_n_calls": self.n_calls,
            }
            if self.episode_rewards:
                summary["final_mean_reward"] = round(float(np.mean(self.episode_rewards[-100:])), 6)
            if self.episode_scores:
                summary["final_mean_score"] = round(float(np.mean(self.episode_scores[-100:])), 6)
            if self.episode_crisprspec:
                summary["final_mean_crisprspec"] = round(float(np.mean(self.episode_crisprspec[-100:])), 6)
                summary["final_mean_gc_content"] = round(float(np.mean(self.episode_gc_content[-100:])), 6)
                summary["final_mean_homopolymer_term"] = round(float(np.mean(self.episode_homopolymer[-100:])), 6)
                summary["final_mean_off_targets"] = round(float(np.mean(self.episode_n_offtargets[-100:])), 2)
            os.makedirs(self.log_dir, exist_ok=True)
            summary_path_json = os.path.join(self.log_dir, "training_summary.json")
            summary_path_txt = os.path.join(self.log_dir, "training_summary.txt")
            with open(summary_path_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            with open(summary_path_txt, "w", encoding="utf-8") as f:
                f.write(f"Training summary\n{'='*50}\n")
                f.write(f"Started:  {summary['training_start']}\n")
                f.write(f"Ended:    {summary['training_end']}\n")
                f.write(f"Duration: {summary['total_time_human']} ({summary['total_time_seconds']} s)\n")
                f.write(f"Timesteps: {summary['total_timesteps']:,}" + (f" (target: {summary['target_timesteps']:,})" if summary.get('target_timesteps') else "") + "\n")
                f.write(f"Episodes: {summary['total_episodes']}\n")
                f.write(f"Status:   {summary['status']}\n")
                if "final_mean_reward" in summary:
                    f.write(f"Final mean reward: {summary['final_mean_reward']}\n")
                if "final_mean_score" in summary:
                    f.write(f"Final mean score:  {summary['final_mean_score']}\n")
                if "final_mean_crisprspec" in summary:
                    f.write(f"Final CRISPRspec: {summary['final_mean_crisprspec']} | GC: {summary['final_mean_gc_content']} | Homopolymer: {summary['final_mean_homopolymer_term']} | Off-targets: {summary['final_mean_off_targets']}\n")
            if self.verbose > 0:
                print(f"  Summary saved: {summary_path_json}, {summary_path_txt}")


def load_sequences(path: str, max_count: int = 1000) -> List[str]:
    """Load gRNA sequences from file. Supports: plain text (one 20-mer per line) or CSV with 'sgRNA' column."""
    import csv
    sequences = []
    path_lower = path.lower()
    if path_lower.endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            sg_col = next((h for h in fieldnames if h.strip().lower() == "sgrna"), None)
            if not sg_col:
                raise ValueError(f"CSV {path} has no 'sgRNA' column (headers: {fieldnames})")
            for row in reader:
                s = (row.get(sg_col) or "").strip().upper().replace("U", "T")
                if len(s) != 20 or not all(c in "ATGC" for c in s):
                    continue
                sequences.append(s)
                if len(sequences) >= max_count:
                    break
        return sequences
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper().replace("U", "T")
            if not s or s.startswith("#"):
                continue
            if len(s) != 20:
                continue
            if not all(c in "ATGC" for c in s):
                continue
            sequences.append(s)
            if len(sequences) >= max_count:
                break
    return sequences


def generate_random_sequences(count: int) -> List[str]:
    """Generate random 20-mer sequences."""
    return ["".join(np.random.choice(BASES) for _ in range(20)) for _ in range(count)]


def make_env(
    sequences: List[str],
    seed_len: int,
    max_steps: int,
    max_mismatches: int = 4,
    min_mismatches: int = 1,
    use_crisprspec: bool = False,
    genome_seq: Optional[str] = None,
    crisprspec_weight: float = 1.0,
    gc_weight: float = 0.1,
    homopolymer_weight: float = 0.1,
    use_cuda_offtarget: bool = True,
    use_dense_reward: bool = True,
    rank: int = 0,
    reference_fasta_path: Optional[str] = None,
):
    """Create a single environment instance. Reward = only CRISPRspec + GC + homopolymer (on_target and energy disabled)."""
    def _init():
        env = CRISPRGymEnv(
            initial_sequences=sequences,
            seed_len=seed_len,
            max_steps=max_steps,
            max_mismatches=max_mismatches,
            min_mismatches=min_mismatches,
            use_crisprspec=use_crisprspec,
            genome_seq=genome_seq,
            crisprspec_weight=crisprspec_weight,
            gc_weight=gc_weight,
            homopolymer_weight=homopolymer_weight,
            # Wire the CLI weights into the actual reward terms. The reward uses the
            # *_penalty_weight fields, so map the user-facing weights onto them.
            off_target_weight=crisprspec_weight,
            gc_penalty_weight=gc_weight,
            homopolymer_penalty_weight=homopolymer_weight,
            use_cuda_offtarget=use_cuda_offtarget,
            reference_fasta_path=reference_fasta_path,
        )
        env = Monitor(env)
        return env
    return _init


def train(
    sequences: List[str],
    total_timesteps: int = 50000,
    seed_len: int = 12,  # Fix last 12 positions (seed region)
    max_steps: int = 10,  # Max actions per episode
    max_mismatches: int = 4,  # Max mutations from original
    min_mismatches: int = 1,  # Min mutations required
    n_envs: int = 1,
    learning_rate: float = 3e-4,
    save_path: str = "RL/models/grna_ppo",
    log_dir: str = "RL/logs",
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
    use_dense_reward: bool = True,  # Dense reward by default (reward every step = score delta)
    ent_coef: float = 0.005,
):
    """
    Train PPO agent on gRNA optimization.
    
    Parameters
    ----------
    sequences : list of str
        Pool of 20-mer sequences to train on.
    total_timesteps : int
        Total training steps.
    seed_len : int
        Fixed positions at start of sequence.
    max_steps : int
        Max steps per episode.
    n_envs : int
        Number of parallel environments.
    learning_rate : float
        PPO learning rate.
    save_path : str
        Path to save trained model.
    log_dir : str
        Directory for tensorboard logs.
    """
    device = "cuda" if (use_cuda and _TORCH_CUDA_AVAILABLE) else "cpu"
    use_cuda_offtarget = use_cuda  # Use CUDA for off-target in env when use_cuda is True

    print(f"Training gRNA RL agent")
    print(f"  Device: {device} (PPO policy)")
    print(f"  Off-target search: {'CUDA (when available)' if use_cuda_offtarget else 'CPU'}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Parallel environments: {n_envs}")
    print()

    # Genome / reference for CRISPRspec: use reference_fasta_path (env loads it) or genome_path (load here)
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
        # Load the reference genome once here and share the same string object across
        # all (training + eval) environments, instead of re-reading the FASTA per env.
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
    print(f"    - Reward: {'dense (every step)' if use_dense_reward else 'sparse (end of episode)'}")
    print(f"    - Entropy coef (exploration): {ent_coef}")
    
    if n_envs == 1:
        env = DummyVecEnv([make_env(sequences, seed_len, max_steps, max_mismatches, min_mismatches, use_crisprspec, genome_seq, crisprspec_weight, gc_weight, homopolymer_weight, use_cuda_offtarget, use_dense_reward, 0, ref_fasta)])
    else:
        env = DummyVecEnv([
            make_env(sequences, seed_len, max_steps, max_mismatches, min_mismatches, use_crisprspec, genome_seq, crisprspec_weight, gc_weight, homopolymer_weight, use_cuda_offtarget, use_dense_reward, i, ref_fasta) for i in range(n_envs)
        ])
    
    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=1024,  # Reduced for more frequent updates
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        verbose=0,  # Reduce SB3 verbosity, we have our own callback
        tensorboard_log=log_dir,
        device=device,
    )
    print(f"  Policy: MlpPolicy (device={device})")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: 64, n_steps: 1024, n_epochs: 10")
    print()
    
    # Training callback - log every 1000 steps; save summary to log_dir at end
    callback = TrainingCallback(verbose=1, log_freq=1000, target_timesteps=total_timesteps, log_dir=log_dir)
    callback.model = model  # Store model reference for callback
    
    # Evaluation callback (if eval_sequences provided)
    callbacks = [callback]
    if eval_sequences:
        print(f"  [Evaluation] Validation set: {len(eval_sequences)} sequences")
        print(f"  [Evaluation] Eval frequency: every {eval_freq} steps")
        print(f"  [Evaluation] Eval episodes: {n_eval_episodes} per evaluation")
        
        eval_env = DummyVecEnv([
            make_env(eval_sequences, seed_len, max_steps, max_mismatches, min_mismatches,
                    use_crisprspec, genome_seq, crisprspec_weight, gc_weight, homopolymer_weight, use_cuda_offtarget, use_dense_reward, 0, ref_fasta)
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
    
    # Train
    print("Starting training...")
    print(f"  Target timesteps: {total_timesteps:,}")
    print(f"  PPO n_steps: 1024, n_envs: {n_envs} → ~{1024 * n_envs} timesteps per update")
    print("-" * 50)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )
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
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    
    if eval_sequences:
        print(f"Best model (based on validation) saved to: {save_path}_best.zip")
    
    # Cleanup
    env.close()
    if eval_sequences and 'eval_env' in locals():
        eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent for gRNA optimization."
    )
    parser.add_argument(
        "--sequences", "-s",
        type=str,
        default=None,
        help="Path to file with training sequences (one 20-mer per line).",
    )
    parser.add_argument(
        "--random-seqs",
        type=int,
        default=0,
        help="Generate N random sequences for training (if --sequences not provided).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Total training timesteps (default: 50000).",
    )
    parser.add_argument(
        "--seed-len",
        type=int,
        default=8,
        help="Fix LAST N positions (seed region near PAM). Default: 8. Mutable region = 0 to (20-N-1).",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=4,
        help="Maximum mismatches from original sequence (default: 4).",
    )
    parser.add_argument(
        "--min-mismatches",
        type=int,
        default=1,
        help="Minimum mismatches required (default: 1).",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=20,
        help="Max steps per episode (default: 20).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "models", "grna_ppo"),
        help="Path to save trained model.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "logs"),
        help="Directory for tensorboard logs.",
    )
    parser.add_argument(
        "--use-crisprspec",
        action="store_true",
        help="Include CRISPRspec surrogate (models/crisproff) in reward for off-target specificity.",
    )
    parser.add_argument(
        "--genome",
        type=str,
        default=None,
        help="Genome FASTA for off-target search (alternative to --reference-fasta).",
    )
    parser.add_argument(
        "--reference-fasta",
        type=str,
        default=None,
        help="Path to reference FASTA for CRISPRspec (e.g. data/genes_with_flankers.fna). Env loads it; overrides --genome.",
    )
    parser.add_argument(
        "--crisprspec-weight",
        type=float,
        default=1,
        help="Weight for CRISPRspec in reward (default: 1).",
    )
    parser.add_argument(
        "--gc-weight",
        type=float,
        default=0.1,
        help="Weight for GC content term in reward; prefer 40-60%% GC (default: 0.1).",
    )
    parser.add_argument(
        "--homopolymer-weight",
        type=float,
        default=0.1,
        help="Weight for homopolymer term in reward; penalize runs > 3 (default: 0.1).",
    )
    parser.add_argument(
        "--eval-sequences",
        type=str,
        default=None,
        help="Path to file with validation sequences (one 20-mer per line). If provided, model will be evaluated on this set during training.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluate model every N steps (default: 5000).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of episodes to run per evaluation (default: 10).",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Use CPU for PPO and for off-target search (default: use GPU when available).",
    )
    parser.add_argument(
        "--no-dense-reward",
        action="store_true",
        help="Use sparse reward (reward only at episode end). Default is dense reward (every step).",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for exploration (default: 0.01). Try 0.05 for more exploration.",
    )
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
        # Default: generate 50 random sequences
        sequences = generate_random_sequences(50)
        print(f"Generated 50 random sequences (use --sequences or --random-seqs to customize)")
    
    if not sequences:
        print("Error: No valid sequences found.")
        sys.exit(1)
    
    # Load evaluation sequences if provided
    eval_sequences = None
    if args.eval_sequences:
        if not os.path.isfile(args.eval_sequences):
            print(f"Warning: Evaluation sequences file not found: {args.eval_sequences}")
            print("         Training will continue without validation.")
        else:
            eval_sequences = load_sequences(args.eval_sequences)
            print(f"Loaded {len(eval_sequences)} validation sequences from {args.eval_sequences}")
    
    # Train
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
        use_dense_reward=not args.no_dense_reward,
        ent_coef=args.ent_coef,
    )


if __name__ == "__main__":
    main()
