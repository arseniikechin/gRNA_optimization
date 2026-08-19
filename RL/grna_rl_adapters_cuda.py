"""
CUDA-accelerated off-target search for gRNA.

Requirements:
  - NVIDIA GPU with CUDA support
  - PyCUDA or Numba with CUDA
  - Recommended: RTX 3060 or better

Usage:
  from RL.grna_rl_adapters_cuda import find_off_targets_in_genome_cuda
  off_targets = find_off_targets_in_genome_cuda(guide_20, genome_seq, max_mismatches=4)
"""

import os
import sys
from typing import List, Optional
import numpy as np

# Add repo root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Try to import CUDA libraries
try:
    from numba import cuda
    import numba
    HAS_NUMBA_CUDA = True
    # Check Numba version for CUDA compatibility
    NUMBA_VERSION = tuple(map(int, numba.__version__.split('.')[:2]))
    # Numba 0.57+ supports CUDA 12.0+ (IR 2.0)
    # Older versions may have IR version conflicts
    CUDA_COMPATIBLE = NUMBA_VERSION >= (0, 57)
except ImportError:
    HAS_NUMBA_CUDA = False
    CUDA_COMPATIBLE = False

try:
    import pycuda.driver as cuda_driver
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

# Fallback to CPU if CUDA not available
if not HAS_NUMBA_CUDA and not HAS_PYCUDA:
    print("[WARNING] CUDA libraries not found. Install numba (pip install numba) or pycuda (pip install pycuda)")
    print("[WARNING] Falling back to CPU implementation.")
    HAS_CUDA = False
else:
    HAS_CUDA = True
    # Log backend details
    if HAS_NUMBA_CUDA:
        try:
            from numba import cuda
            if cuda.is_available():
                print(f"[CUDA] Using Numba CUDA (device: {cuda.get_current_device().name})")
            else:
                print("[CUDA] Numba installed but CUDA not available (will try PyCUDA or CPU)")
        except:
            pass
    # If Numba is available, disable PyCUDA to avoid context conflicts
    if HAS_NUMBA_CUDA:
        HAS_PYCUDA = False  # Use only Numba CUDA
    elif HAS_PYCUDA:
        try:
            import pycuda.driver as cuda_driver
            # Do not import autoinit at module import time (can break Numba context handling)
            # import pycuda.autoinit  # Deferred import only when using PyCUDA implementation
            print(f"[CUDA] PyCUDA available (CUDA version: {cuda_driver.get_version()})")
        except (ImportError, TypeError, SyntaxError) as e:
            print(f"[CUDA] PyCUDA incompatible: {type(e).__name__}")
            HAS_PYCUDA = False
        except Exception as e:
            print(f"[CUDA] PyCUDA error: {e}")

# Reverse complement mapping
_REV_MAP = str.maketrans("ACGTacgt", "TGCAtgca")

# Genome caching on GPU
_genome_cache = {}  # {cache_key: (d_genome_device_array, genome_len)}
_offtarget_results_cache = {}  # {(guide_20, genome_hash): [off_target_23mers]}
_cuda_device_initialized = False


def _rev_comp(seq: str) -> str:
    """Reverse complement of DNA sequence."""
    return seq.translate(_REV_MAP)[::-1]


# ============================================================================
# Numba CUDA implementation (preferred - easier to use)
# ============================================================================

if HAS_NUMBA_CUDA:
    # Use simpler syntax without explicit signatures for better compatibility
    @cuda.jit
    def _cuda_find_offtargets_plus_strand(
        genome_array, guide_array, pam_val0, pam_val1, max_mismatches, 
        exclude_array, results_array, results_count, genome_len
    ):
        idx = cuda.grid(1)
        if idx >= genome_len - 22:
            return
        
        # PAM check
        if genome_array[idx + 21] != pam_val0 or genome_array[idx + 22] != pam_val1:
            return
        
        # Hamming distance
        mismatches = 0
        for i in range(20):
            if genome_array[idx + i] != guide_array[i]:
                mismatches += 1
                if mismatches > max_mismatches:
                    return
        
        # Save result
        result_idx = cuda.atomic.add(results_count, 0, 1)
        if result_idx < results_array.shape[0]:
            results_array[result_idx] = idx
    
    @cuda.jit
    def _cuda_find_offtargets_minus_strand(
        genome_array, guide_array, pam_rc_val0, pam_rc_val1, max_mismatches,
        exclude_array, results_array, results_count, genome_len
    ):
        idx = cuda.grid(1)
        if idx >= genome_len - 22:
            return
        
        # PAM check: a real NGG site on the minus strand reads as "CCN..." on the
        # plus strand, so the window must start with "CC" (encoded 1, 1). The two
        # G's of the PAM map to these two C positions; the N maps to idx+2.
        if genome_array[idx] != pam_rc_val0 or genome_array[idx + 1] != pam_rc_val1:
            return
        
        # Hamming distance for reverse complement of the spacer (genome idx+3..idx+22).
        # A base encoded as 4 ('N'/unknown) yields rev_base = -1, never matching A/C/G/T.
        mismatches = 0
        for i in range(20):
            pos_in_genome = idx + 22 - i
            rev_base = 3 - genome_array[pos_in_genome]
            if rev_base != guide_array[i]:
                mismatches += 1
                if mismatches > max_mismatches:
                    return
        
        # Save result
        result_idx = cuda.atomic.add(results_count, 0, 1)
        if result_idx < results_array.shape[0]:
            results_array[result_idx] = idx


def _encode_dna_to_array(seq: str) -> np.ndarray:
    """Encode DNA sequence to uint8 array: A=0, C=1, G=2, T=3, N/unknown=4.

    Unknown bases map to 4 (a sentinel) so they never match a real A/C/G/T in the
    kernels and never satisfy the PAM check, avoiding spurious hits in N regions
    and at record junctions.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([mapping.get(c, 4) for c in seq.upper()], dtype=np.uint8)


def _decode_array_to_dna(arr: np.ndarray) -> str:
    """Decode uint8 array back to a DNA string."""
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    return ''.join(mapping.get(int(x), 'N') for x in arr)


def find_off_targets_in_genome_cuda(
    guide_20: str,
    genome_seq: str = None,
    max_mismatches: int = 4,
    pam_suffixes: tuple = ("GG",),
    exclude_23: Optional[str] = None,
    threads_per_block: int = 256,
    gene_id: Optional[str] = None,
    fasta_path: Optional[str] = None,
    flank_size: int = 100_000_000,
) -> List[str]:
    """
    CUDA-accelerated search for off-target sites in a genome.
    """
    if not HAS_CUDA:
        if not hasattr(find_off_targets_in_genome_cuda, '_cpu_warned'):
            print("[Off-target search] CUDA libraries not available, using CPU (SLOW!)")
            find_off_targets_in_genome_cuda._cpu_warned = True
        from RL.grna_rl_adapters import find_off_targets_in_genome
        return find_off_targets_in_genome(guide_20, genome_seq, max_mismatches, pam_suffixes, exclude_23)
    
    if HAS_NUMBA_CUDA:
        try:
            from numba import cuda
            if not cuda.is_available():
                raise RuntimeError("Numba CUDA not available")
        except Exception as e:
            if not hasattr(find_off_targets_in_genome_cuda, '_cuda_import_error'):
                print(f"[Off-target search] Cannot import/use Numba CUDA: {e}")
                print("[Off-target search] Falling back to CPU implementation")
                find_off_targets_in_genome_cuda._cuda_import_error = True
            from RL.grna_rl_adapters import find_off_targets_in_genome
            return find_off_targets_in_genome(guide_20, genome_seq, max_mismatches, pam_suffixes, exclude_23)
    else:
        if not hasattr(find_off_targets_in_genome_cuda, '_no_numba_warned'):
            print("[Off-target search] Numba CUDA not available, using CPU")
            find_off_targets_in_genome_cuda._no_numba_warned = True
        from RL.grna_rl_adapters import find_off_targets_in_genome
        return find_off_targets_in_genome(guide_20, genome_seq, max_mismatches, pam_suffixes, exclude_23)
    
    if not hasattr(find_off_targets_in_genome_cuda, '_logged'):
        print(f"[Off-target search] Using Numba CUDA on {cuda.get_current_device().name}")
        find_off_targets_in_genome_cuda._logged = True
    
    guide_20 = guide_20.upper()[:20]
    if len(guide_20) != 20:
        return []
    
    # Check results cache (keyed by guide + exclude)
    cache_key_seq = (guide_20, exclude_23)
    if cache_key_seq in _offtarget_results_cache:
        return _offtarget_results_cache[cache_key_seq]
    
    # Prepare genome — cache the uppercased version and GPU array by object id + length
    # (avoids re-encoding and MD5 hashing on every call)
    import time as time_module
    
    genome_cache_key = (id(genome_seq), len(genome_seq))
    
    if not hasattr(find_off_targets_in_genome_cuda, '_genome_cache'):
        find_off_targets_in_genome_cuda._genome_cache = {}
    if not hasattr(find_off_targets_in_genome_cuda, '_genome_str_cache'):
        find_off_targets_in_genome_cuda._genome_str_cache = {}
    
    # Cache the uppercased genome string (avoid 10MB .upper() on every call)
    if genome_cache_key not in find_off_targets_in_genome_cuda._genome_str_cache:
        genome_upper = genome_seq.upper().replace("U", "T")
        find_off_targets_in_genome_cuda._genome_str_cache[genome_cache_key] = genome_upper
    else:
        genome_upper = find_off_targets_in_genome_cuda._genome_str_cache[genome_cache_key]
    
    n = len(genome_upper)
    if n < 23:
        return []
    
    # Encode sequences
    guide_array = _encode_dna_to_array(guide_20)
    
    # PAM values as scalars (GG = 2, 2)
    pam_encoded = _encode_dna_to_array(pam_suffixes[0])[:2]
    pam_val0, pam_val1 = int(pam_encoded[0]), int(pam_encoded[1])
    pam_rc_val0, pam_rc_val1 = 1, 1  # CC for minus strand
    
    # Exclude array
    if exclude_23:
        exclude_array = _encode_dna_to_array(exclude_23.upper()[:23])
    else:
        exclude_array = np.array([], dtype=np.uint8)
    
    # Max results: with 4 mismatches on a 20-mer, typical off-targets are <10k even in large genomes
    # 500k is a safe upper bound; allocates ~2MB GPU memory per strand
    max_results = 500000
    
    # Cache genome on GPU (avoid re-transfer)
    if genome_cache_key not in find_off_targets_in_genome_cuda._genome_cache:
        try:
            transfer_start = time_module.time()
            genome_array = _encode_dna_to_array(genome_upper)
            d_genome = cuda.to_device(genome_array)
            transfer_time = time_module.time() - transfer_start
            find_off_targets_in_genome_cuda._genome_cache[genome_cache_key] = d_genome
            print(f"[CUDA] Genome transfer to GPU: {transfer_time:.2f} s ({n/1e6:.1f} Mbp) [NEW TRANSFER]")
        except Exception as e:
            if not hasattr(find_off_targets_in_genome_cuda, '_cuda_error_warned'):
                print(f"[Off-target search] CUDA error during genome transfer: {e}")
                print("[Off-target search] Falling back to CPU implementation")
                find_off_targets_in_genome_cuda._cuda_error_warned = True
            from RL.grna_rl_adapters import find_off_targets_in_genome
            return find_off_targets_in_genome(guide_20, genome_upper, max_mismatches, pam_suffixes, exclude_23)
    else:
        d_genome = find_off_targets_in_genome_cuda._genome_cache[genome_cache_key]
        if not hasattr(find_off_targets_in_genome_cuda, '_cache_used_logged'):
            print("[CUDA] Using cached genome on GPU")
            find_off_targets_in_genome_cuda._cache_used_logged = True
    
    d_guide = cuda.to_device(guide_array)
    d_exclude = cuda.to_device(exclude_array)
    
    d_results_plus = cuda.device_array(max_results, dtype=np.int32)
    d_results_minus = cuda.device_array(max_results, dtype=np.int32)
    d_count_plus = cuda.to_device(np.array([0], dtype=np.int32))
    d_count_minus = cuda.to_device(np.array([0], dtype=np.int32))
    
    min_blocks = 1
    total_positions = n - 22
    blocks_per_grid = max(min_blocks, (total_positions + threads_per_block - 1) // threads_per_block)
    
    # Launch kernels
    try:
        kernel_start = time_module.time()
        _cuda_find_offtargets_plus_strand[blocks_per_grid, threads_per_block](
            d_genome, d_guide, pam_val0, pam_val1, max_mismatches,
            d_exclude, d_results_plus, d_count_plus, n
        )
        _cuda_find_offtargets_minus_strand[blocks_per_grid, threads_per_block](
            d_genome, d_guide, pam_rc_val0, pam_rc_val1, max_mismatches,
            d_exclude, d_results_minus, d_count_minus, n
        )
        kernel_time = time_module.time() - kernel_start
        if kernel_time > 0.1:
            print(f"[CUDA] Kernel execution: {kernel_time:.2f} s ({n/1e6:.1f} Mbp, {blocks_per_grid} blocks)")
    except Exception as e:
        error_msg = str(e)
        if "IR_VERSION_MISMATCH" in error_msg or "NVVM_ERROR" in error_msg or "Failed to compile" in error_msg:
            print(f"[WARNING] CUDA kernel compilation error: {error_msg}")
            print("[INFO] Falling back to CPU implementation.")
            from RL.grna_rl_adapters import find_off_targets_in_genome
            return find_off_targets_in_genome(guide_20, genome_upper, max_mismatches, pam_suffixes, exclude_23)
        else:
            raise
    
    # Sync and collect results
    cuda.synchronize()
    count_plus = d_count_plus.copy_to_host()[0]
    count_minus = d_count_minus.copy_to_host()[0]
    
    if count_plus >= max_results or count_minus >= max_results:
        print(f"[WARNING CUDA] Result buffer overflow! Plus: {count_plus}/{max_results}, Minus: {count_minus}/{max_results}")
    
    results = []
    
    # Extract exclude spacer for comparison (first 20bp — matches any PAM variant)
    exclude_spacer = exclude_23.upper()[:20] if exclude_23 and len(exclude_23) >= 20 else None
    
    # Plus strand results
    if count_plus > 0:
        results_plus = d_results_plus[:min(count_plus, max_results)].copy_to_host()
        for start in results_plus:
            if start >= 0 and start + 23 <= n:
                seq = genome_upper[start:start + 23]
                if len(seq) == 23 and (not exclude_spacer or seq[:20] != exclude_spacer):
                    results.append(seq)
    
    # Minus strand results
    if count_minus > 0:
        results_minus = d_results_minus[:min(count_minus, max_results)].copy_to_host()
        for start in results_minus:
            if start >= 0 and start + 23 <= n:
                w = genome_upper[start:start + 23]
                w_rc = _rev_comp(w)
                if len(w_rc) == 23 and (not exclude_spacer or w_rc[:20] != exclude_spacer):
                    results.append(w_rc)
    
    # Deduplicate
    results_unique = list(dict.fromkeys(results))
    
    # Cache result (limit cache size)
    if len(_offtarget_results_cache) < 5000:
        _offtarget_results_cache[cache_key_seq] = results_unique
    
    return results_unique


# ============================================================================
# Fallback: CPU implementation (if CUDA not available)
# ============================================================================

def find_off_targets_in_genome_cuda_fallback(
    guide_20: str,
    genome_seq: str,
    max_mismatches: int = 4,
    pam_suffixes: tuple = ("GG",),
    exclude_23: Optional[str] = None,
) -> List[str]:
    """Fallback CPU implementation when CUDA is unavailable."""
    from RL.grna_rl_adapters import find_off_targets_in_genome
    return find_off_targets_in_genome(guide_20, genome_seq, max_mismatches, pam_suffixes, exclude_23)


# ============================================================================
# Main function with auto-detection
# ============================================================================

def find_off_targets_in_genome_cuda_auto(
    guide_20: str,
    genome_seq: str,
    max_mismatches: int = 4,
    pam_suffixes: tuple = ("GG",),
    exclude_23: Optional[str] = None,
    force_cpu: bool = False,
) -> List[str]:
    """
    Automatically choose CUDA or CPU implementation.

    Parameters
    ----------
    force_cpu : bool
        Force CPU even when CUDA is available.
    """
    if force_cpu or not HAS_CUDA:
        return find_off_targets_in_genome_cuda_fallback(
            guide_20, genome_seq, max_mismatches, pam_suffixes, exclude_23
        )
    
    return find_off_targets_in_genome_cuda(
        guide_20, genome_seq, max_mismatches, pam_suffixes, exclude_23
    )


# ============================================================================
# Test and benchmark function
# ============================================================================

def benchmark_cuda_vs_cpu(
    guide_20: str = "GTCCTACAGGTATGGATCTC",
    genome_size: int = 1000000,  # 1 Mbp for testing
    iterations: int = 3,
):
    """Compare CUDA vs CPU performance."""
    import time
    import random
    
    # Generate test genome
    bases = "ACGT"
    genome_seq = ''.join(random.choice(bases) for _ in range(genome_size))
    
    print(f"Benchmark: guide={guide_20}, genome={genome_size} bp")
    print(f"GPU: {cuda.get_current_device().name if HAS_CUDA else 'N/A'}")
    print()
    
    # CPU benchmark
    from RL.grna_rl_adapters import find_off_targets_in_genome
    cpu_times = []
    for i in range(iterations):
        start = time.time()
        cpu_results = find_off_targets_in_genome(guide_20, genome_seq, max_mismatches=4)
        cpu_times.append(time.time() - start)
    cpu_avg = np.mean(cpu_times)
    print(f"CPU: {cpu_avg:.3f} s (avg), {len(cpu_results)} off-targets")
    
    # CUDA benchmark
    if HAS_CUDA:
        cuda_times = []
        for i in range(iterations):
            start = time.time()
            cuda_results = find_off_targets_in_genome_cuda(guide_20, genome_seq, max_mismatches=4)
            cuda_times.append(time.time() - start)
        cuda_avg = np.mean(cuda_times)
        print(f"CUDA: {cuda_avg:.3f} s (avg), {len(cuda_results)} off-targets")
        print(f"Speedup: {cpu_avg / cuda_avg:.2f}x")
    else:
        print("CUDA: Not available")


if __name__ == "__main__":
    # Test
    test_guide = "GTCCTACAGGTATGGATCTC"
    test_genome = "A" * 1000 + "GTCCTACAGGTATGGATCTCTGG" + "T" * 1000 + "GTCCTACAGGTATGGATCTCTGG" + "C" * 1000
    
    print("Testing CUDA off-target search...")
    if HAS_CUDA:
        results = find_off_targets_in_genome_cuda(test_guide, test_genome, max_mismatches=4)
        print(f"Found {len(results)} off-targets")
        print(f"Results: {results[:5]}")
    else:
        print("CUDA not available, using CPU fallback")
        results = find_off_targets_in_genome_cuda_fallback(test_guide, test_genome, max_mismatches=4)
        print(f"Found {len(results)} off-targets (CPU)")
