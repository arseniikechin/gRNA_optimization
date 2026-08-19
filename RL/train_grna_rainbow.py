"""
Train a Rainbow DQN agent for gRNA optimization.

Same environment / reward as the PPO and QR-DQN scripts (CRISPRspec off-target + GC +
homopolymer, dense reward every step), but the policy is a self-contained Rainbow DQN
implemented in PyTorch. Rainbow combines six DQN extensions:

  1. Double DQN            - online net selects the next action, target net evaluates it
  2. Dueling architecture  - separate value and advantage streams
  3. Prioritized replay    - samples transitions proportionally to TD error
  4. Multi-step returns    - n-step bootstrapping
  5. Distributional (C51)  - learns a categorical value distribution over atoms
  6. NoisyNets             - learned noisy layers replace epsilon-greedy exploration

There is no Rainbow implementation in stable-baselines3 / sb3-contrib, so this script
does not use SB3 for the agent. It only depends on PyTorch (already required) and reuses
the existing CRISPRGymEnv and sequence loaders.

Example (Linux/macOS), run from RL/:
    python train_grna_rainbow.py \
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
import time
import json
import math
from collections import deque
from typing import List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("Error: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

from RL.grna_gym_env import CRISPRGymEnv
from RL.train_grna_rl import load_sequences, generate_random_sequences


# ---------------------------------------------------------------------------
# Gym/Gymnasium compatibility helpers
# ---------------------------------------------------------------------------
def env_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
    else:
        obs, reward, done, info = out
        terminated, truncated = done, False
    return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# NoisyNet linear layer (factorized Gaussian noise)
# ---------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ---------------------------------------------------------------------------
# Dueling + distributional (C51) network with noisy layers
# ---------------------------------------------------------------------------
class RainbowNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, n_atoms: int, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )
        # Value stream
        self.value_hidden = NoisyLinear(hidden, hidden)
        self.value = NoisyLinear(hidden, n_atoms)
        # Advantage stream
        self.adv_hidden = NoisyLinear(hidden, hidden)
        self.adv = NoisyLinear(hidden, n_actions * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability distribution over atoms, shape (batch, n_actions, n_atoms)."""
        f = self.feature(x)
        v = self.value(F.relu(self.value_hidden(f))).view(-1, 1, self.n_atoms)
        a = self.adv(F.relu(self.adv_hidden(f))).view(-1, self.n_actions, self.n_atoms)
        q_atoms = v + a - a.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=2)
        return dist.clamp(min=1e-6)

    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value.reset_noise()
        self.adv_hidden.reset_noise()
        self.adv.reset_noise()


# ---------------------------------------------------------------------------
# Prioritized experience replay (proportional, array-based)
# ---------------------------------------------------------------------------
class PrioritizedReplay:
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.5):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.disc = np.zeros(capacity, dtype=np.float32)  # gamma ** n_steps_used
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def add(self, s, a, r, ns, done, disc):
        i = self.pos
        self.states[i] = s
        self.next_states[i] = ns
        self.actions[i] = a
        self.rewards[i] = r
        self.dones[i] = float(done)
        self.disc[i] = disc
        self.priorities[i] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        prios = self.priorities[: self.size] ** self.alpha
        probs = prios / prios.sum()
        idx = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[idx]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return idx, weights

    def update_priorities(self, idx, prios):
        prios = np.abs(prios) + 1e-6
        self.priorities[idx] = prios
        self.max_priority = max(self.max_priority, float(prios.max()))


# ---------------------------------------------------------------------------
# Environment factory (reward wiring identical to the fixed make_env)
# ---------------------------------------------------------------------------
def build_env(sequences, seed_len, max_steps, max_mismatches, min_mismatches,
              use_crisprspec, genome_seq, crisprspec_weight, gc_weight, homopolymer_weight,
              use_cuda_offtarget, reference_fasta_path):
    return CRISPRGymEnv(
        initial_sequences=sequences,
        seed_len=seed_len,
        max_steps=max_steps,
        max_mismatches=max_mismatches,
        min_mismatches=min_mismatches,
        use_crisprspec=use_crisprspec,
        genome_seq=genome_seq,
        crisprspec_weight=crisprspec_weight,
        off_target_weight=crisprspec_weight,
        gc_weight=gc_weight,
        homopolymer_weight=homopolymer_weight,
        gc_penalty_weight=gc_weight,
        homopolymer_penalty_weight=homopolymer_weight,
        use_cuda_offtarget=use_cuda_offtarget,
        reference_fasta_path=reference_fasta_path,
    )


def resolve_genome(use_crisprspec, reference_fasta_path, genome_path):
    """Load the genome once (shared object). Returns (genome_seq, ref_fasta_for_env)."""
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
            print(f"  WARNING: --use-crisprspec specified but genome not loaded! CRISPRspec will be 0.")
    elif use_crisprspec and ref_fasta:
        from RL.grna_rl_adapters import load_genome
        genome_seq = load_genome(ref_fasta)
        print(f"  [CRISPRspec] Genome loaded once: {len(genome_seq)} bp from {ref_fasta}")
        ref_fasta = None
    return genome_seq, ref_fasta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    sequences: List[str],
    total_timesteps: int = 50000,
    seed_len: int = 8,
    max_steps: int = 20,
    max_mismatches: int = 4,
    min_mismatches: int = 1,
    learning_rate: float = 1e-4,
    save_path: str = "RL/models/grna_rainbow",
    log_dir: str = "RL/logs_rainbow",
    use_crisprspec: bool = False,
    genome_path: Optional[str] = None,
    reference_fasta_path: Optional[str] = None,
    crisprspec_weight: float = 1.0,
    gc_weight: float = 0.1,
    homopolymer_weight: float = 0.1,
    use_cuda: bool = True,
    # Rainbow hyperparameters
    buffer_size: int = 100_000,
    learning_starts: int = 1000,
    batch_size: int = 32,
    gamma: float = 0.99,
    n_step: int = 3,
    target_update_interval: int = 8000,
    train_freq: int = 1,
    hidden: int = 128,
    n_atoms: int = 51,
    v_min: float = -10.0,
    v_max: float = 10.0,
    per_alpha: float = 0.5,
    per_beta0: float = 0.4,
    adam_eps: float = 1.5e-4,
    log_freq: int = 1000,
):
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    use_cuda_offtarget = use_cuda

    print("Training gRNA RL agent (Rainbow DQN)")
    print(f"  Device: {device}")
    print(f"  Off-target search: {'CUDA (when available)' if use_cuda_offtarget else 'CPU'}")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Max steps per episode: {max_steps}")

    genome_seq, ref_fasta = resolve_genome(use_crisprspec, reference_fasta_path, genome_path)

    print(f"  Mutable region: positions 0-{20-seed_len-1} ({20-seed_len} positions)")
    print(f"  Seed region (fixed): positions {20-seed_len}-19 ({seed_len} positions)")
    print(f"  Mismatches allowed: {min_mismatches}-{max_mismatches}")
    print(f"  Reward: CRISPRspec {'ENABLED' if (use_crisprspec and (genome_seq or ref_fasta)) else 'DISABLED'} "
          f"(w={crisprspec_weight}) + GC (w={gc_weight}) + homopolymer (w={homopolymer_weight})")
    print(f"  Rainbow: atoms={n_atoms} [{v_min},{v_max}], n_step={n_step}, PER(alpha={per_alpha},beta0={per_beta0})")
    print(f"  Buffer: {buffer_size}, batch: {batch_size}, target_update: {target_update_interval}, exploration: NoisyNets")

    env = build_env(sequences, seed_len, max_steps, max_mismatches, min_mismatches,
                    use_crisprspec, genome_seq, crisprspec_weight, gc_weight, homopolymer_weight,
                    use_cuda_offtarget, ref_fasta)

    n_actions = env.action_space.n
    obs_dim = int(np.prod(env.observation_space.shape))

    online = RainbowNet(obs_dim, n_actions, n_atoms, hidden).to(device)
    target = RainbowNet(obs_dim, n_actions, n_atoms, hidden).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate, eps=adam_eps)
    replay = PrioritizedReplay(buffer_size, obs_dim, alpha=per_alpha)

    support = torch.linspace(v_min, v_max, n_atoms, device=device)
    delta_z = (v_max - v_min) / (n_atoms - 1)

    def act(obs_np: np.ndarray) -> int:
        online.reset_noise()
        with torch.no_grad():
            state = torch.as_tensor(obs_np.reshape(1, -1), dtype=torch.float32, device=device)
            dist = online(state)  # (1, A, atoms)
            q = (dist * support).sum(dim=2)  # (1, A)
            return int(q.argmax(dim=1).item())

    def learn(beta: float):
        idx, weights = replay.sample(batch_size, beta)
        states = torch.as_tensor(replay.states[idx], device=device)
        next_states = torch.as_tensor(replay.next_states[idx], device=device)
        actions = torch.as_tensor(replay.actions[idx], device=device)
        rewards = torch.as_tensor(replay.rewards[idx], device=device).unsqueeze(1)
        dones = torch.as_tensor(replay.dones[idx], device=device).unsqueeze(1)
        disc = torch.as_tensor(replay.disc[idx], device=device).unsqueeze(1)
        w = torch.as_tensor(weights, device=device)
        batch_idx = torch.arange(batch_size, device=device)

        with torch.no_grad():
            online.reset_noise()
            next_dist_online = online(next_states)
            next_q = (next_dist_online * support).sum(dim=2)
            next_actions = next_q.argmax(dim=1)  # Double DQN action selection

            target.reset_noise()
            next_dist = target(next_states)
            next_dist = next_dist[batch_idx, next_actions]  # (B, atoms)

            Tz = rewards + (1.0 - dones) * disc * support.unsqueeze(0)
            Tz = Tz.clamp(v_min, v_max)
            b = (Tz - v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            # Handle l == u so probability is not lost
            l[(u > 0) & (l == u)] -= 1
            u[(l < (n_atoms - 1)) & (l == u)] += 1

            m = torch.zeros(batch_size, n_atoms, device=device)
            offset = (torch.arange(batch_size, device=device) * n_atoms).unsqueeze(1)
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        online.reset_noise()
        dist = online(states)
        log_p = torch.log(dist[batch_idx, actions].clamp(min=1e-8))  # (B, atoms)
        loss_per = -(m * log_p).sum(dim=1)  # (B,)
        loss = (loss_per * w).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), 10.0)
        optimizer.step()

        replay.update_priorities(idx, loss_per.detach().cpu().numpy())
        return float(loss.item())

    # n-step buffer
    nstep_buf = deque(maxlen=n_step)

    def push_nstep(final: bool = False):
        """Emit n-step transitions from the deque into the replay buffer."""
        while nstep_buf:
            if not final and len(nstep_buf) < n_step:
                break
            R = 0.0
            used = 0
            done_flag = False
            next_state = nstep_buf[-1][3]
            for k, (s, a, r, ns, d) in enumerate(nstep_buf):
                R += (gamma ** k) * r
                used = k + 1
                next_state = ns
                if d:
                    done_flag = True
                    break
            s0, a0 = nstep_buf[0][0], nstep_buf[0][1]
            replay.add(s0, a0, R, next_state, done_flag, gamma ** used)
            nstep_buf.popleft()
            if not final:
                break

    os.makedirs(log_dir, exist_ok=True)
    online.train()

    episode_rewards: List[float] = []
    episode_reward = 0.0
    ep_count = 0
    start_time = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    last_print = start_time

    obs = env_reset(env)
    print("Starting training...\n" + "-" * 50)

    for step in range(1, total_timesteps + 1):
        obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        action = act(obs_flat)
        next_obs, reward, terminated, truncated, info = env_step(env, action)
        next_flat = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        # Bootstrap only on true terminal (terminated), not on time-limit (truncated)
        nstep_buf.append((obs_flat, action, float(reward), next_flat, bool(terminated)))
        push_nstep(final=False)

        obs = next_obs
        episode_reward += float(reward)

        if terminated or truncated:
            push_nstep(final=True)
            episode_rewards.append(episode_reward)
            ep_count += 1
            episode_reward = 0.0
            obs = env_reset(env)

        # Anneal PER beta from beta0 -> 1
        beta = min(1.0, per_beta0 + (1.0 - per_beta0) * step / total_timesteps)

        loss_val = None
        if replay.size >= max(batch_size, learning_starts) and step % train_freq == 0:
            loss_val = learn(beta)

        if step % target_update_interval == 0:
            target.load_state_dict(online.state_dict())

        # Logging
        if step % log_freq == 0 or step == 1 or time.time() - last_print > 10:
            last_print = time.time()
            elapsed = time.time() - start_time
            sps = step / elapsed if elapsed > 0 else 0.0
            recent = episode_rewards[-50:]
            mean_r = float(np.mean(recent)) if recent else float("nan")
            pct = 100.0 * step / total_timesteps
            print(f"[{step:,}/{total_timesteps:,} {pct:4.1f}%] eps={ep_count} "
                  f"reward(last{len(recent)})={mean_r:.4f} "
                  f"loss={loss_val if loss_val is not None else float('nan'):.4f} "
                  f"buffer={replay.size} beta={beta:.3f} {sps:.1f} steps/s {elapsed:.0f}s")
            sys.stdout.flush()

    print("-" * 50)
    print("Training complete!")

    # Save model + config for a matching inference script
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt_path = save_path + ".pt"
    torch.save({
        "model_state": online.state_dict(),
        "config": {
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "n_atoms": n_atoms,
            "hidden": hidden,
            "v_min": v_min,
            "v_max": v_max,
            "seed_len": seed_len,
            "max_steps": max_steps,
            "max_mismatches": max_mismatches,
        },
    }, ckpt_path)
    print(f"Model saved to {ckpt_path}")

    # Lightweight training summary
    elapsed = time.time() - start_time
    summary = {
        "algo": "rainbow_dqn",
        "training_start": start_iso,
        "training_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_seconds": round(elapsed, 2),
        "total_timesteps": total_timesteps,
        "total_episodes": ep_count,
        "final_mean_reward": round(float(np.mean(episode_rewards[-100:])), 6) if episode_rewards else None,
    }
    with open(os.path.join(log_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved: {os.path.join(log_dir, 'training_summary.json')}")

    env.close()
    return online


def main():
    parser = argparse.ArgumentParser(description="Train Rainbow DQN agent for gRNA optimization.")
    parser.add_argument("--sequences", "-s", type=str, default=None,
                        help="Path to training sequences (one 20-mer per line, or CSV with sgRNA column).")
    parser.add_argument("--random-seqs", type=int, default=0,
                        help="Generate N random sequences (if --sequences not provided).")
    parser.add_argument("--steps", type=int, default=50000, help="Total training timesteps.")
    parser.add_argument("--seed-len", type=int, default=8, help="Fix LAST N positions (seed region). Default 8.")
    parser.add_argument("--max-mismatches", type=int, default=4, help="Max mismatches from original.")
    parser.add_argument("--min-mismatches", type=int, default=1, help="Min mismatches required.")
    parser.add_argument("--max-episode-steps", type=int, default=20, help="Max steps per episode.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--save-path", type=str, default=os.path.join(_SCRIPT_DIR, "models", "grna_rainbow"),
                        help="Path to save trained model (without extension; .pt is appended).")
    parser.add_argument("--log-dir", type=str, default=os.path.join(_SCRIPT_DIR, "logs_rainbow"),
                        help="Directory for logs / training summary.")
    parser.add_argument("--use-crisprspec", action="store_true", help="Include CRISPRspec off-target in reward.")
    parser.add_argument("--genome", type=str, default=None, help="Genome FASTA (alternative to --reference-fasta).")
    parser.add_argument("--reference-fasta", type=str, default=None, help="Reference FASTA for CRISPRspec; overrides --genome.")
    parser.add_argument("--crisprspec-weight", type=float, default=1.0, help="Weight for CRISPRspec in reward.")
    parser.add_argument("--gc-weight", type=float, default=0.1, help="Weight for GC penalty.")
    parser.add_argument("--homopolymer-weight", type=float, default=0.1, help="Weight for homopolymer penalty.")
    parser.add_argument("--force-cpu", action="store_true", help="Use CPU for the policy and off-target search.")
    # Rainbow-specific
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size.")
    parser.add_argument("--learning-starts", type=int, default=1000, help="Steps before learning starts.")
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--n-step", type=int, default=3, help="Multi-step return length.")
    parser.add_argument("--target-update-interval", type=int, default=8000, help="Steps between target updates.")
    parser.add_argument("--train-freq", type=int, default=1, help="Gradient update every N steps.")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer size.")
    parser.add_argument("--n-atoms", type=int, default=51, help="Number of C51 atoms.")
    parser.add_argument("--v-min", type=float, default=-10.0, help="Minimum return support.")
    parser.add_argument("--v-max", type=float, default=10.0, help="Maximum return support.")
    parser.add_argument("--per-alpha", type=float, default=0.5, help="Prioritized replay alpha.")
    parser.add_argument("--per-beta0", type=float, default=0.4, help="Prioritized replay initial beta.")
    args = parser.parse_args()

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

    train(
        sequences=sequences,
        total_timesteps=args.steps,
        seed_len=args.seed_len,
        max_steps=args.max_episode_steps,
        max_mismatches=args.max_mismatches,
        min_mismatches=args.min_mismatches,
        learning_rate=args.lr,
        save_path=args.save_path,
        log_dir=args.log_dir,
        use_crisprspec=args.use_crisprspec,
        genome_path=args.genome,
        reference_fasta_path=args.reference_fasta,
        crisprspec_weight=args.crisprspec_weight,
        gc_weight=args.gc_weight,
        homopolymer_weight=args.homopolymer_weight,
        use_cuda=not args.force_cpu,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        n_step=args.n_step,
        target_update_interval=args.target_update_interval,
        train_freq=args.train_freq,
        hidden=args.hidden,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        per_alpha=args.per_alpha,
        per_beta0=args.per_beta0,
    )


if __name__ == "__main__":
    main()
