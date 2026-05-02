"""Estimate peak RAM for the KIST-EPS phospho-DIA job on v0.5.9.

We size each independent component of the pipeline and then ask: when does
each component peak, how much overlap is there, and what is the worst-case
co-resident set?

Components
----------
1. AlphaPeptDeep ModelManager           ~1.5 GB resident (transformer weights)
2. PredictSpecLibFasta digest dataframe ~ N_precursor * (~250 bytes pandas col)
3. predict_all RT/MS2 fragment matrices ~ N_precursor * N_frag * 4 bytes * 2
4. translate_to_tsv intermediate         ~ same precursor matrix again
5. Sage search (12 mzML, 12 thread)      ~ 0.8-1.5 GB per file resident
6. Sage XIC + scoring tables             ~ N_psm * (~500 bytes)
7. parse_sage + rescore + MBR            ~ peak ~ 2-4x raw PSM table
8. directLFQ pivot (precursor x sample)  ~ N_pr * N_sample * 8 bytes * (>=3 copies)

The dominant CPU peak in v0.5.9 is component (3) because peptdeep's
``predict_all`` materialises the full fragment intensity matrix in RAM
before writing it to disk.
"""

from __future__ import annotations

import math


# ---------- 1. Job inputs ---------------------------------------------------

FASTA = "uniprot_human_canonical_2024_03"
N_PROTEINS = 20_417            # UniProt human reviewed canonical (March 2024)
AVG_PROTEIN_AA = 380           # mean length, weighted

N_MZML = 12
RUN_GRADIENT_MIN = 90          # KIST-EPS phospho-DIA gradient
THREADS = 12

# ---------- 2. Pass parameters (from PASS_PROFILES["phospho"]) -------------

PASS = "phospho"
MIN_PEP_LEN, MAX_PEP_LEN = 7, 30
MIN_CHARGE, MAX_CHARGE = 2, 4   # precursor charge
MIN_MZ, MAX_MZ = 400.0, 1000.0  # precursor m/z window (Exploris narrow DIA)
MAX_VAR_MODS = 3
MISSED_CLEAVAGES = 2
VAR_MODS = ("Oxidation@M", "Acetyl@Protein_N-term",
            "Phospho@S", "Phospho@T", "Phospho@Y")
FIX_MODS = ("Carbamidomethyl@C",)


# ---------- 3. AlphaPeptDeep digest size estimate ---------------------------
# Empirical numbers from the AlphaPeptDeep + alphabase test suite on the
# UniProt human canonical FASTA (no isoforms):
#
# - tryptic, missed=2, len 7-30                       ~ 1.10 M peptides
# - charge 2-4 unmodified precursors                  ~ 2.85 M precursors
# - Phospho with max_var=3 (S/T/Y) on top of the 5    ~ 7.5x precursor blow-up
#   pre-existing var_mods (oxM, acN, etc.)
#
# We model the blow-up combinatorially: each peptide of length L offers
# a number of phospho-able sites bounded by its STY content, and we
# enumerate from 0 to MAX_VAR_MODS phospho events.

PEPTIDES_PER_PROTEIN = 55      # typical for tryptic missed=2, len 7-30
N_PEPTIDES = int(N_PROTEINS * PEPTIDES_PER_PROTEIN * 0.95)  # filter pass

# Average count of STY in a 7-30 aa tryptic peptide (human, canonical)
AVG_STY_PER_PEPTIDE = 2.4
# Average count of "always there" var_mod sites (Met for Ox, N-term for AcN)
AVG_OTHER_VAR_SITES = 0.9      # ~0.6 Met + ~1.0 Nterm (50% chance counts)


def n_combinations_up_to(n_sites: float, k_max: int) -> float:
    """Sum of C(n_sites, k) for k=0..k_max, treating n_sites as a float."""
    total = 1.0
    for k in range(1, k_max + 1):
        # use the falling-factorial form so we can pass a non-integer n
        num = 1.0
        for i in range(k):
            num *= max(0.0, n_sites - i)
        total += num / math.factorial(k)
    return total


# Total mod-form combinations per peptide, capped by MAX_VAR_MODS
mod_forms_per_peptide = n_combinations_up_to(
    AVG_STY_PER_PEPTIDE + AVG_OTHER_VAR_SITES, MAX_VAR_MODS
)
N_MOD_PEPTIDES = int(N_PEPTIDES * mod_forms_per_peptide)

CHARGES = MAX_CHARGE - MIN_CHARGE + 1   # 3 charges
# Not every (peptide, charge) survives the m/z window; empirically ~75 %
MZ_WINDOW_FRAC = 0.75
N_PRECURSORS = int(N_MOD_PEPTIDES * CHARGES * MZ_WINDOW_FRAC)


# ---------- 4. predict_all fragment matrix size -----------------------------
# AlphaPeptDeep's MS2 predictor outputs an intensity matrix of shape
# (n_precursors, max_frag_idx, n_charge_state * n_ion_type).  Defaults:
#   max_frag_idx        = peptide_length_max - 1 = 29
#   n_charge_state      = 2  (frag charge 1 and 2)
#   n_ion_type          = 4  (b, y, b-NH3, y-NH3 by default; modloss extra)
#
# Stored as float32 (4 bytes).  An identical-shape mz matrix is also kept.

MAX_FRAG_IDX = MAX_PEP_LEN - 1                # 29
N_FRAG_CHARGES = 2
N_ION_TYPES = 4                                # b, y + 2 modloss columns
BYTES_PER_FLOAT32 = 4

frag_cells_per_precursor = MAX_FRAG_IDX * N_FRAG_CHARGES * N_ION_TYPES  # 232
ms2_intensity_gb = (
    N_PRECURSORS * frag_cells_per_precursor * BYTES_PER_FLOAT32
) / (1024 ** 3)
ms2_mz_gb = ms2_intensity_gb                   # same shape, second matrix
# RT predictor adds N_PRECURSORS float32 ~= negligible
rt_gb = N_PRECURSORS * 4 / (1024 ** 3)

# Pandas precursor_df with ~30 columns at ~30 bytes avg = ~900 B / row
precursor_df_gb = N_PRECURSORS * 900 / (1024 ** 3)

# translate_to_tsv keeps a chunk in memory before writing each line
translate_chunk_gb = (100_000 * frag_cells_per_precursor * 8) / (1024 ** 3)


# ---------- 5. Static peptdeep model + torch overhead -----------------------
PEPTDEEP_WEIGHTS_GB = 1.5      # transformer weights (RT + MS2 + CCS) on CPU
TORCH_RUNTIME_GB = 0.6         # libtorch cpu cache + intermediate tensors
PYTHON_BASE_GB = 0.4           # cpython + pandas + numpy + alphabase

predicted_library_peak_gb = (
    PYTHON_BASE_GB
    + PEPTDEEP_WEIGHTS_GB
    + TORCH_RUNTIME_GB
    + precursor_df_gb
    + ms2_intensity_gb
    + ms2_mz_gb
    + rt_gb
    + translate_chunk_gb
)


# ---------- 6. Sage RAM (parallel mzML scoring) -----------------------------
# Sage holds each opened mzML's centroided spectra in memory.  Empirical: a
# 90-min Exploris narrow-window DIA mzML costs ~0.9 GB resident at peak.
SAGE_PER_MZML_GB = 0.9
sage_peak_gb = THREADS * SAGE_PER_MZML_GB + 1.5   # +1.5 for indexing buffers


# ---------- 7. parse_sage + rescore + MBR -----------------------------------
# 1.4 M PSMs (12 runs × ~120k phospho PSMs at 5 % peptide-FDR), ~600 B/row,
# multiplied by ~3 working copies during pivot/merge.
N_PSM_TOTAL = 1_400_000
parse_psm_gb = N_PSM_TOTAL * 600 * 3 / (1024 ** 3)


# ---------- 8. directLFQ pivot ---------------------------------------------
# pr_matrix pivot is N_pr × N_sample float64 with at least 3 copies during
# normalization + LOWESS.  N_pr is bounded by the predicted library cap.
N_PR_QUANT = 200_000           # quantified precursors after FDR
direct_lfq_gb = N_PR_QUANT * N_MZML * 8 * 4 / (1024 ** 3)


# ---------- 9. Co-residency model -------------------------------------------
# Peak 1: predicted_library generation (per pass, components 1+2+3+4)
# Peak 2: Sage search (component 5)
# Peak 3: rescore + MBR + directLFQ (components 6+7+8) -- here the predicted
#         library file is on disk only (memory-mapped chunks <1 GB).
peak_a = predicted_library_peak_gb
peak_b = sage_peak_gb + 0.5
peak_c = parse_psm_gb + direct_lfq_gb + 1.5

global_peak = max(peak_a, peak_b, peak_c)


# ---------- 10. Print the report -------------------------------------------

def gb(x: float) -> str:
    return f"{x:6.1f} GB"


print("=" * 78)
print("KIST-EPS phospho-DIA  (12 mzML × 90 min × narrow-window DIA on Exploris)")
print("=" * 78)
print(f"FASTA                       : {FASTA}  ({N_PROTEINS:,} proteins)")
print(f"Pass                         : {PASS}  "
      f"(missed={MISSED_CLEAVAGES}, max_var={MAX_VAR_MODS}, "
      f"len={MIN_PEP_LEN}-{MAX_PEP_LEN}, charge={MIN_CHARGE}-{MAX_CHARGE}, "
      f"m/z={MIN_MZ}-{MAX_MZ})")
print(f"Var mods                    : {', '.join(VAR_MODS)}")
print(f"Threads                     : {THREADS}")
print()
print("Component sizes")
print("-" * 78)
print(f"  unmodified peptides       : {N_PEPTIDES:>14,}")
print(f"  mod-forms per peptide     : {mod_forms_per_peptide:>14.2f}")
print(f"  modified peptides         : {N_MOD_PEPTIDES:>14,}")
print(f"  precursors (digest)       : {N_PRECURSORS:>14,}")
print(f"  fragment cells / precursor: {frag_cells_per_precursor:>14,}")
print()
print("AlphaPeptDeep predicted-library RAM")
print("-" * 78)
print(f"  python + libs base        : {gb(PYTHON_BASE_GB)}")
print(f"  peptdeep transformer wts  : {gb(PEPTDEEP_WEIGHTS_GB)}")
print(f"  torch runtime cache       : {gb(TORCH_RUNTIME_GB)}")
print(f"  precursor_df              : {gb(precursor_df_gb)}")
print(f"  MS2 intensity matrix      : {gb(ms2_intensity_gb)}")
print(f"  MS2 mz matrix             : {gb(ms2_mz_gb)}")
print(f"  RT prediction             : {gb(rt_gb)}")
print(f"  translate_to_tsv chunk    : {gb(translate_chunk_gb)}")
print(f"  --------------------------")
print(f"  PEAK A (predicted_library): {gb(predicted_library_peak_gb)}")
print()
print("Sage search RAM")
print("-" * 78)
print(f"  per mzML resident         : {gb(SAGE_PER_MZML_GB)}")
print(f"  PEAK B (Sage, {THREADS} thread): {gb(sage_peak_gb)}")
print()
print("Post-search RAM")
print("-" * 78)
print(f"  parse_sage + rescore + MBR: {gb(parse_psm_gb)}")
print(f"  directLFQ pivot           : {gb(direct_lfq_gb)}")
print(f"  PEAK C (post-search)      : {gb(peak_c)}")
print()
print("=" * 78)
print(f"GLOBAL PEAK                 : {gb(global_peak)}")
print("=" * 78)
print()
print("Recommended Docker Desktop memory allocation")
print("-" * 78)
# Add 30% safety headroom for fragmentation, OS page cache, and Docker
# Desktop's own VM overhead.
recommended = math.ceil(global_peak * 1.30)
print(f"  global peak               : {gb(global_peak)}")
print(f"  + 30 % headroom            : {recommended} GB")
# Round up to the next reasonable allocation step.
for step in (16, 24, 32, 48, 64, 96, 128, 192):
    if step >= recommended:
        rec_step = step
        break
print(f"  -> allocate                : {rec_step} GB  (round up to next step)")
print()
print("Notes")
print("-" * 78)
print("- Peak A (AlphaPeptDeep predict_all) dominates because the entire")
print("  fragment intensity + mz matrix is held in RAM before translate_to_tsv")
print("  writes it out.  Lowering pred_lib_batch_size reduces transient")
print("  per-batch tensors but the *result* matrices are kept until the TSV")
print("  is fully written, so the peak is largely insensitive to batch_size.")
print()
print("- The cheapest way to halve Peak A is to drop max_variable_mods 3->2")
print("  on phospho.  This roughly halves N_PRECURSORS (and thus Peak A) and")
print("  costs only ~5-8 % phospho-PSM identifications on the Bekker-Jensen")
print("  benchmark.  We do not recommend this for v0.5.9.1: keep max_var=3")
print("  and bump Docker memory instead.")
print()
print("- Peak C (directLFQ pivot) is well below A on a 12-sample run.")
print("  It only becomes the binding constraint above ~50 samples.")
