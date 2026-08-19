"""
Adapters for gRNA RL environment: genome loading, off-target search, CRISPRspec surrogate.
Uses energy/CRISPRspec_CRISPRoff_pipeline.py for CRISPRspec scoring.
"""

import os
import sys
from typing import List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reverse complement for DNA (PAM / off-target on both strands)
_REV_MAP = str.maketrans("ACGTacgt", "TGCAtgca")


def _rev_comp(seq: str) -> str:
    """Reverse complement of DNA sequence."""
    return seq.translate(_REV_MAP)[::-1]


def find_pam_in_genome(guide_20: str, genome_seq: str, pam_suffixes: tuple = ("GG",)) -> Optional[str]:
    """
    Find the real 3-letter PAM for a 20-mer guide in the genome.
    Searches both strands for exact match (0 mismatches) with valid PAM.
    Returns 3-letter PAM (e.g. "TGG", "AGG") or None if not found.
    """
    guide_20 = guide_20.upper()[:20]
    if len(guide_20) != 20:
        return None
    genome = genome_seq.upper().replace("U", "T")
    n = len(genome)
    
    # Plus strand: [20 spacer][3 PAM], last 2 of PAM must be in pam_suffixes
    for i in range(n - 22):
        w = genome[i : i + 23]
        if "N" in w:
            continue
        if w[-2:] not in pam_suffixes:
            continue
        if w[:20] == guide_20:
            return w[20:23]  # 3-letter PAM
    
    # Minus strand: a real NGG site on the minus strand reads as "CCN..." on the
    # plus strand, so after reverse-complementing the window its PAM ends with "GG".
    pam_suffixes_rc = ("GG",)
    for i in range(n - 22):
        w = genome[i : i + 23]
        if "N" in w:
            continue
        w_rc = _rev_comp(w)
        if w_rc[-2:] not in pam_suffixes_rc:
            continue
        if w_rc[:20] == guide_20:
            return w_rc[20:23]
    
    return None


def load_genome(path: str) -> str:
    """
    Load genome from FASTA file; return single concatenated sequence (uppercase).

    Records are joined with a run of 'N' (>= one full 23-mer window) so that no
    guide/PAM window can span two records and create spurious off-target hits at
    the junction. Off-target search skips any window containing 'N'.
    """
    from Bio import SeqIO
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for record in SeqIO.parse(f, "fasta"):
            seqs.append(str(record.seq).upper().replace("U", "T"))
    return ("N" * 23).join(seqs)


def find_off_targets_in_genome(
    guide_20: str,
    genome_seq: str,
    max_mismatches: int = 4,
    pam_suffixes: tuple = ("GG",),
    exclude_23: Optional[str] = None,
) -> List[str]:
    """
    Find off-target sites (23-mer: 20 spacer + 3 PAM) in genome for a 20-mer guide.
    Searches both strands. Returns list of 23-mer strings (spacer + PAM).
    Considers only PAM GG (plus strand) / CC (minus strand). Includes 0–max_mismatches;
    if exclude_23 is given, that 23-mer (the on-target gRNA site) is not included.
    """
    guide_20 = guide_20.upper()[:20]
    if len(guide_20) != 20:
        return []
    genome = genome_seq.upper().replace("U", "T")
    n = len(genome)
    results = []
    # GG-only window. A real NGG site on the minus strand reads as "CCN..." on the
    # plus strand, so after reverse-complementing the window its PAM ends with "GG".
    pam_suffixes_rc = ("GG",)
    
    # Extract exclude spacer (first 20bp) for comparison — matches any PAM variant
    exclude_spacer = None
    if exclude_23 is not None:
        exclude_spacer = exclude_23.upper()[:20]

    def hamming(a: str, b: str) -> int:
        return sum(1 for x, y in zip(a, b) if x != y)

    def add_if_not_self(w23: str) -> None:
        # Exclude on-target by spacer match (0 mismatches in first 20bp)
        # This correctly excludes the on-target regardless of PAM variant
        if exclude_spacer and len(w23) >= 20 and w23[:20] == exclude_spacer:
            return
        results.append(w23)

    # + strand: genome layout [20 spacer][3 PAM], PAM ends with GG
    for i in range(n - 22):
        w = genome[i : i + 23]
        if len(w) != 23:
            continue
        if "N" in w:
            continue
        if w[-2:] not in pam_suffixes:
            continue
        if 0 <= hamming(guide_20, w[:20]) <= max_mismatches:
            add_if_not_self(w)

    # - strand: minus-strand NGG reads as "CCN..." on the plus strand; after rev_comp
    # the window becomes [20 spacer][3 PAM] with PAM ending in GG.
    for i in range(n - 22):
        w = genome[i : i + 23]
        if "N" in w:
            continue
        w_rc = _rev_comp(w)
        if w_rc[-2:] not in pam_suffixes_rc:
            continue
        if 0 <= hamming(guide_20, w_rc[:20]) <= max_mismatches:
            add_if_not_self(w_rc)

    return list(dict.fromkeys(results))


def predict_crisprspec_surrogate(
    guide_20: str,
    off_targets_23: List[str],
    use_grna_folding: bool = False,
    pam_3: Optional[str] = None,
) -> Optional[float]:
    """
    Compute CRISPRspec surrogate score using energy/CRISPRspec_CRISPRoff_pipeline.
    guide_20: 20-mer gRNA spacer.
    off_targets_23: list of 23-mer off-target sequences (20 + PAM).
    pam_3: 3-letter PAM (e.g. "TGG", "AGG"). If None, uses "GGG" so guide_23 = 23-mer.
    Returns score in ~[0, 10] (higher = more specific). Returns None on error.
    """
    import math
    guide_20 = guide_20.upper()[:20]
    if len(guide_20) != 20:
        return None

    try:
        from energy.CRISPRspec_CRISPRoff_pipeline import (
            read_energy_parameters,
            compute_CRISPRspec,
            calcRNADNAenergy,
        )
    except ImportError:
        try:
            sys.path.insert(0, _REPO_ROOT)
            from energy.CRISPRspec_CRISPRoff_pipeline import (
                read_energy_parameters,
                compute_CRISPRspec,
                calcRNADNAenergy,
            )
        except ImportError:
            return None

    energy_pkl = os.path.join(_REPO_ROOT, "energy", "energy_dics.pkl")
    if not os.path.isfile(energy_pkl):
        return None
    # Load energy parameters only once (pkl read is slow)
    if not getattr(predict_crisprspec_surrogate, '_energy_loaded', False):
        read_energy_parameters(energy_pkl)
        predict_crisprspec_surrogate._energy_loaded = True

    # 23-mer = 20 spacer + 3 PAM (previously +"GG" gave 22-mer and caused failures with off-targets)
    if pam_3 and len(pam_3) == 3 and all(c in "ACGT" for c in pam_3.upper()):
        guide_23 = guide_20 + pam_3.upper()
    else:
        guide_23 = guide_20 + "GGG"
    if len(guide_23) != 23:
        return None
    
    # Filter and validate off-targets: must be exactly 23-mer, only ACGT (no N or other IUPAC)
    # If genome contains N, all off-targets may be filtered out; use site-count heuristic instead of fixed 10.0
    valid_off_targets = []
    for ot in off_targets_23:
        ot_upper = ot.upper()
        if len(ot_upper) == 23 and all(c in "ACGT" for c in ot_upper):
            valid_off_targets.append(ot_upper)
    
    if not valid_off_targets:
        # No valid off-targets (e.g., all contain N). Use site-count-based score so metric still varies
        if not off_targets_23:
            return 10.0
        n = len(off_targets_23)
        return max(0.0, 10.0 - min(9.0, math.log10(1 + n)))
    
    ontarget = (guide_23, guide_23, "", 0, 0, "")
    off_seqs = [(ot, "") for ot in valid_off_targets]

    try:
        # Filter off-targets: ensure they have same length as guide_23 after RI_REV_NT_MAP transformation
        from energy.CRISPRspec_CRISPRoff_pipeline import RI_REV_NT_MAP
        filtered_off_seqs = []
        for ot_seq, chrom in off_seqs:
            if len(ot_seq) != len(guide_23):
                continue
            # Check transformed length - must match guide_23 length
            transformed = ''.join([RI_REV_NT_MAP.get(c, '') for c in ot_seq])
            if len(transformed) == len(guide_23):
                filtered_off_seqs.append((ot_seq, chrom))
        
        if not filtered_off_seqs:
            # No off-targets remain after RI_REV_NT_MAP filtering; score by number of input sites
            if not valid_off_targets:
                return 10.0
            n = len(valid_off_targets)
            return max(0.0, 10.0 - min(9.0, math.log10(1 + n)))
        
        on_prob, _ = compute_CRISPRspec(
            ontarget,
            filtered_off_seqs,
            calcRNADNAenergy,
            GU_allowed=False,
            pos_weight=True,
            pam_corr=True,
            grna_folding=use_grna_folding,
            dna_opening=True,
            dna_pos_wgh=False,
            ignored_chromosomes=set(),
        )
    except Exception as e:
        import sys
        sys.stderr.write(f"[predict_crisprspec_surrogate] Error: {e}\n")
        sys.stderr.write(f"  guide_23={guide_23} (len={len(guide_23)})\n")
        sys.stderr.write(f"  off_seqs count={len(off_seqs)}\n")
        if off_seqs:
            sys.stderr.write(f"  first off_seq={off_seqs[0][0]} (len={len(off_seqs[0][0])})\n")
        import traceback
        sys.stderr.write(traceback.format_exc())
        return None

    # compute_CRISPRspec returns P_off (off-target probability). Higher means lower specificity.
    # score = -log10(P_off), capped at 10. To avoid saturation at high off-target counts,
    # apply a small penalty based on the number of off-targets.
    if on_prob <= 0:
        return 10.0
    raw = -math.log10(on_prob)
    penalty = min(2.0, math.log10(1 + len(filtered_off_seqs)) * 0.5)
    return max(0.0, min(10.0, raw - penalty))


def compute_hybridization_energy_single(guide_20: str) -> Optional[float]:
    """
    Compute hybridization energy (deltaG) for on-target: 20-mer guide vs self (perfect match).
    Used when use_energy=True in the env. Returns float or None on error.
    """
    guide_20 = guide_20.upper()[:20]
    if len(guide_20) != 20:
        return None
    try:
        from energy.CRISPRspec_CRISPRoff_pipeline import (
            read_energy_parameters,
            get_eng,
            calcRNADNAenergy,
        )
    except ImportError:
        try:
            sys.path.insert(0, _REPO_ROOT)
            from energy.CRISPRspec_CRISPRoff_pipeline import (
                read_energy_parameters,
                get_eng,
                calcRNADNAenergy,
            )
        except ImportError:
            return None
    energy_pkl = os.path.join(_REPO_ROOT, "energy", "energy_dics.pkl")
    if not os.path.isfile(energy_pkl):
        return None
    read_energy_parameters(energy_pkl)
    guide_23 = guide_20 + "GG"
    try:
        dg = get_eng(
            guide_23,
            guide_23,
            calcRNADNAenergy,
            GU_allowed=False,
            pos_weight=True,
            pam_corr=True,
            grna_folding=False,
            dna_opening=True,
            dna_pos_wgh=False,
        )
    except Exception:
        return None
    return float(dg)
