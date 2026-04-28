# diaquant

**Open-source DIA proteomics quantification with universal PTM support.**

`diaquant` is a Python pipeline that takes converted **mzML** files from a Thermo Orbitrap (or Bruker timsTOF) DIA experiment and produces **DIA-NN-compatible** quantification matrices for both intact proteins and post-translationally modified sites. Unlike DIA-NN, every component is fully open-source and the pipeline natively supports phosphorylation, ubiquitylation, acetylation, mono/di/tri-methylation, succinylation, malonylation, crotonylation, SUMO/GG remnants and arbitrary user-defined modifications without re-training.

The whole stack is permissively licensed (Apache-2.0 / MIT) and can therefore be deployed in commercial pipelines without restriction.

## Architecture

| Stage | Component | Licence | Role |
|------|-----------|---------|------|
| 1. mzML parsing & search | [Sage](https://github.com/lazear/sage) v0.14 (Rust) | MIT | Spectrum-centric DIA search with arbitrary variable mods, target-decoy FDR |
| 2. PTM site localization | `diaquant.ptm_localization` | Apache-2.0 | Softmax over Sage hyperscores → site probability |
| 3. Quantification | [directLFQ](https://github.com/MannLabs/directlfq) v0.3 | Apache-2.0 | MaxLFQ-equivalent rollup, in O(n) time |
| 4. RT alignment | `diaquant.rt_align` | Apache-2.0 | LOWESS run-to-run retention-time alignment (always-on) |
| 5. Output | `diaquant.writer` | Apache-2.0 | DIA-NN-style `report.pr_matrix.tsv`, `report.pg_matrix.tsv`, plus PTM-site matrix and RT-alignment statistics |

The optional `[deeplearning]` extra installs **AlphaPeptDeep** (Apache-2.0) to enable on-the-fly RT/MS² prediction and transfer learning for non-canonical PTMs (introduced in 0.5.0).

## Installation

```bash
# 1. install the Sage binary (≈ 9 MB)
wget https://github.com/lazear/sage/releases/download/v0.14.7/sage-v0.14.7-x86_64-unknown-linux-gnu.tar.gz
tar xzf sage-v0.14.7-x86_64-unknown-linux-gnu.tar.gz
sudo cp sage-v0.14.7-x86_64-unknown-linux-gnu/sage /usr/local/bin/

# 2. install diaquant itself
cd /path/to/diaquant
pip install -e .
# optional: pip install -e .[deeplearning]
```

## Quick start (single-pass)

```bash
# 1. generate a starter YAML for a single Sage search with selected PTMs
diaquant init-config \
    --output configs/my_run.yaml \
    --fasta uniprot_mouse.fasta \
    --mzml-dir ./mzml/ \
    --ptm Phospho --ptm Acetyl --ptm Methyl --ptm GlyGly --ptm Succinyl

# 2. run the full pipeline
diaquant run --config configs/my_run.yaml
```

## Multi-pass workflow (recommended for whole-proteome PTM analysis)

When analysing **non-enriched whole-proteome** mzML files, running every PTM together in one Sage search multiplies the index size and degrades sensitivity for *every* PTM family. `diaquant` 0.3 introduces a **multi-pass** mode: each PTM family is searched in its own Sage pass with PTM-specific parameter overrides (e.g. `missed_cleavages=3` for K-acyl mods because the modification blocks tryptic cleavage at lysine), and the resulting precursor tables are merged before quantification.

```bash
# 1. list available passes
diaquant list-passes

# 2. generate a config that runs only the passes you want
diaquant init-config \
    --output configs/my_run.yaml \
    --fasta uniprot_mouse.fasta \
    --mzml-dir ./mzml/ \
    --pass whole_proteome --pass phospho --pass acetyl_methyl \
    --sample-sheet configs/sample_sheet_KIST_EPS.tsv

# 3. run the pipeline; each pass executes Sage independently
diaquant run --config configs/my_run.yaml
```

Built-in passes:

| Pass | Variable mods | `missed_cleavages` | Why |
|---|---|---|---|
| `whole_proteome` | `Oxidation`, `Acetyl_Nterm` | 2 | Backbone for protein-group quantification |
| `phospho` | + `Phospho` (STY) | 2 | Site probability cut-off 0.75, length 7–30 |
| `ubiquitin` | + `GlyGly` (K) | 3 | GG remnant blocks tryptic K cleavage |
| `acetyl_methyl` | + `Acetyl`, `Methyl`, `Dimethyl`, `Trimethyl` | 3 | Acyl/methyl K mods block cleavage; **DIA-NN cannot quantify these reliably**. Grouped together because they compete for the same K residues and are often studied together. |
| `succinyl_acyl` | + `Succinyl`, `Malonyl`, `Crotonyl` | 3 | Same K-acyl rationale |
| `oglcnac` (0.5.0) | + `OGlcNAc` (ST) | 2 | O-GlcNAc does not block tryptic cleavage, so `missed_cleavages` stays at 2. Fragment tolerance relaxed to 15 ppm for sugar loss. |
| `citrullination` (0.5.0) | + `Citrullination` (R) | 2 | Isobaric with deamidation (+0.984 Da); requires high MS1 accuracy. |
| `lactyl_acyl` (0.5.0) | + `Lactyl`, `Propionyl`, `Butyryl` | 3 | Less common K-acylations, grouped because they all block tryptic cleavage at K. |

### Why are some PTMs grouped together (e.g., Methyl + Acetyl)?
In multi-pass mode, PTMs that compete for the same residue (like K-acetylation and K-methylation) or share the same enzymatic cleavage behavior (like all K-acylations blocking trypsin) are grouped into a single pass. This allows the search engine to correctly assign the modification when multiple isobaric or near-isobaric possibilities exist on the same peptide, without blowing up the search space for unrelated PTMs (like phosphorylation).

### How enzyme rules are handled
By default, diaquant uses `trypsin` (cleaves at K/R, except when followed by P). The `missed_cleavages` parameter is handled at two levels:
1. **Global default**: Set in the main config (usually 2).
2. **Pass-specific override**: PTMs that structurally block the protease (like K-acetylation blocking trypsin) automatically override the global setting for their specific pass (e.g., `missed_cleavages = 3`).

User-defined passes can be added under `custom_passes:` in YAML — see `configs/preset_wholeproteome_multipass.yaml` for an example. Selecting only the passes you need keeps the run focused and the search space small.

## Ready-made presets

The `configs/` directory ships with four drop-in presets tuned for **Thermo Orbitrap Exploris 240 with 6 m/z DIA windows**, which matches the acquisition method of the included KIST-EPS data set:

| Preset | Use case | Notes |
|--------|----------|-------|
| `preset_phospho_exploris240.yaml` | STY-phosphorylation enrichment (single-pass) | Mirrors the user's `phosphoidentificationoption.txt` |
| `preset_ubiquitin_exploris240.yaml` | K-GG (di-glycyl) ubiquitin remnant (single-pass) | Mirrors `ubiquitylationidentificationoption.txt`, `missed_cleavages = 3` |
| `preset_multiPTM_exploris240.yaml` | All PTMs in one Sage search (legacy) | Larger index, kept for reference |
| **`preset_wholeproteome_multipass.yaml`** | **Whole-proteome protein expression + selectable PTM passes (recommended)** | Edit the `passes:` list to control which PTM families are searched |

## DIA-NN ↔ diaquant parameter map

Every field in DIA-NN's GUI has an equivalent in the diaquant YAML so you can migrate identical settings:

| DIA-NN GUI field | diaquant YAML key | Default |
|------------------|-------------------|---------|
| Protease | `enzyme` | `trypsin` |
| Missed cleavages | `missed_cleavages` | `2` |
| Peptide length | `min_peptide_length`, `max_peptide_length` | `7`, `30` |
| Precursor charge | `min_precursor_charge`, `max_precursor_charge` | `2`, `4` |
| Precursor m/z | `min_precursor_mz`, `max_precursor_mz` | `400`, `1000` |
| Fragment ion m/z | `min_fragment_mz`, `max_fragment_mz` | `200`, `1500` |
| MS1 accuracy (ppm) | `precursor_tol_ppm` | `6` |
| MS2 accuracy (ppm) | `fragment_tol_ppm` | `12` |
| Library MS1 / MS2 (ppm) | `library_precursor_tol_ppm`, `library_fragment_tol_ppm` | `8`, `15` |
| Variable modifications | `variable_modifications` | — |
| Max var modifications | `max_variable_mods` | `2` |
| Scoring | `scoring_mode` | `peptidoforms` |
| MBR | `match_between_runs` | `true` |
| Quantification | (always QuantUMS-equivalent via directLFQ) | — |
| NNs cross-validated | `machine_learning` | `nn_cv` |

## Notes on Thermo Orbitrap Exploris 240 (and other narrow-window DIA instruments)

The defaults above are tuned to the actual acquisition window observed in the user's data: 100 staggered 6 m/z windows centered between **403 and ≈ 1000 m/z**, MS1 every ~50 MS2 scans. Three settings in the original DIA-NN parameter files were adjusted in the diaquant presets to better match this acquisition:

1. **`min_precursor_mz` raised from 350 to 400** — DIA-NN was set to start at 350 m/z but the lowest isolation centre is 403 m/z, so peptides under 400 m/z were never sampled.
2. **`max_peptide_length` and `max_precursor_charge` unified** between identification and library generation. The original Phospho library YAML used 35 / 6 vs. the identification's 30 / 4, which silently dropped library precursors at search time.
3. **`max_variable_mods` left at 2 for identification, 3 for the multi-PTM library**. DIA-NN's library file allowed 4, which doubles the search space for marginal sensitivity gain on Exploris 240.

All of these are configurable; use `init-config --help` to see the full list.

## Outputs

All files are written to `output_dir` and are tab-separated, UTF-8.

| File | DIA-NN equivalent | Description |
|------|------------------|-------------|
| `report.pr_matrix.tsv` | `report.pr_matrix.tsv` | One row per precursor, one column per mzML file. Columns 1–10 are identical to DIA-NN. |
| `report.pg_matrix.tsv` | `report.pg_matrix.tsv` | One row per protein group (directLFQ), columns 1–6 identical to DIA-NN. |
| `report.ptm_site_matrix.tsv` | *(none)* | One row per `(gene, residue+position, PTM type)` triple, quantified across all runs with directLFQ. |
| `report.tsv` | `report.tsv` | Long-format master table with PTM site probabilities, per-pass origin and FDR fields. |
| `report.pg_matrix.diff.tsv` | *(none)* | Pairwise log2 fold-change, Welch t-test p-value and BH q-value between every group pair defined in the sample sheet (only written when `sample_sheet:` is set). |
| `report.rt_alignment.tsv` | *(none)* | One row per mzML file with anchor count, drift median/IQR/RMSE in seconds **before and after** LOWESS alignment, and the maximum absolute correction applied. |

In multi-pass mode each pass also writes its own raw Sage outputs to `output_dir/pass_<name>/`, so individual searches can be inspected or re-quantified in isolation.

The first ten columns of `report.pr_matrix.tsv` are: `Protein.Group`, `Protein.Ids`, `Protein.Names`, `Genes`, `First.Protein.Description`, `Proteotypic`, `Stripped.Sequence`, `Modified.Sequence`, `Precursor.Charge`, `Precursor.Id`. The remaining columns are the input mzML file paths, exactly as DIA-NN writes them.

## Run-to-run retention time alignment (LOWESS, always-on)

LC retention times routinely drift between samples — a few seconds is normal, tens of seconds is not unusual after column ageing or temperature changes. Even modest drift causes the same peptide to fall into a different DIA cycle, increasing missing values and degrading site-level quantification. `diaquant` 0.4 therefore aligns every run to a common reference using **LOWESS** (the same approach used by MaxQuant and Skyline), and this is enabled by default.

The alignment runs after the Sage search and before quantification. The run with the largest number of confident PSMs (q ≤ `rt_align_q_cutoff`) is chosen as the reference, then for every other run a LOWESS curve is fitted on the shared anchor peptides (matched by `Modified.Sequence` + `Precursor.Charge`) and applied to all PSMs in that run. The original `RT` column is preserved and a new `RT.Aligned` column is appended to `report.tsv`.

Drift statistics are written to `report.rt_alignment.tsv`, with one row per mzML file and the following columns: `filename`, `role` (`reference`, `aligned`, or `skipped (too few anchors)`), `n_anchors`, `drift_median_sec_before`, `drift_iqr_sec_before`, `rmse_sec_before`, `drift_median_sec_after`, `drift_iqr_sec_after`, `rmse_sec_after`, `max_abs_correction_sec`. Comparing the before/after RMSE columns gives an immediate, quantitative picture of how much drift was corrected — typically a 5–10× reduction on real data. The CLI also prints the median before/after RMSE on the console for a quick sanity check.

All behaviour is controlled by four YAML keys: `rt_alignment` (default `true`; set to `false` to disable), `rt_align_frac` (LOWESS smoothing fraction, default `0.2`), `rt_align_min_anchors` (minimum shared peptides required, default `50`; runs with fewer anchors are reported but not modified) and `rt_align_q_cutoff` (PSM q-value upper bound for anchor selection, default `0.01`).

## Universal PTM support

DIA-NN's MiniDNN was trained primarily on phosphorylation and GlyGly data, so non-canonical PTMs (acetylation, mono/di/tri-methylation, succinylation, …) suffer from poor fragment-intensity prediction, missing site localization and unsupported protein-C-terminal modifications. `diaquant` sidesteps these issues in three ways:

1. **Search-engine independence from any PTM-trained NN.** Sage matches fragments by m/z; arbitrary mass shifts are accepted without retraining.
2. **Site-level directLFQ instead of Top-1 quantification.** DIA-NN reports the single strongest peptidoform per site; `diaquant` aggregates every supporting precursor with the directLFQ ratio model.
3. **AlphaPeptDeep predicted spectral libraries (New in 0.5.0).** diaquant automatically generates a full-proteome predicted spectral library for *any* combination of PTMs using AlphaPeptDeep's transformer models. The predicted RT is then used to rescore Sage PSMs (`Pred.RT.Delta`), heavily penalizing false positives for rare PTMs.
4. **Transfer learning (New in 0.5.0).** Opt-in fine-tuning of the AlphaPeptDeep RT/MS2 models using the high-confidence PSMs from your own first pass, adapting the predictions to your exact LC gradient and collision energy.
5. **One-line custom PTMs.** Add an entry under `custom_modifications:` in the YAML — no recompilation, no model fine-tuning needed. If AlphaPeptDeep knows the UniMod name, it will even predict the RT/MS2 correctly:
   ```yaml
   custom_modifications:
     - name: ADP_Ribosyl
       unimod_id: 213
       mass_shift: 541.06111
       targets: [D, E, R, K]
   ```

## Sample sheet and differential analysis

Provide a TSV with at least two columns to enable automatic group comparisons:

```tsv
mzml_file	condition	timepoint	replicate	group
EPS_NonECM_0h_1.mzML	NonECM_EPS	0h	1	NonECM_EPS_0h
EPS_NonECM_0h_2.mzML	NonECM_EPS	0h	2	NonECM_EPS_0h
... (one row per mzML file)
```

The `group` column drives all pairwise comparisons; `condition`, `timepoint` and `replicate` are recorded for downstream plotting. A complete 36-sample template for the KIST-EPS data set is included as `configs/sample_sheet_KIST_EPS.tsv`.

When the sample sheet is referenced from the YAML (`sample_sheet: configs/sample_sheet_KIST_EPS.tsv`), `diaquant run` automatically writes `report.pg_matrix.diff.tsv` containing log2 fold-change, Welch t-test p-value and Benjamini-Hochberg q-value for every group pair.

## Roadmap

Future versions will expand the AlphaPeptDeep integration to replace the softmax-based site localization with a predicted-MS2 Δ-score, and use the AlphaPeptDeep predictions as a *third* anchor source for RT alignment, allowing accurate alignment even for runs whose shared-PSM count falls below `rt_align_min_anchors`.

## Licence

Apache-2.0. See `LICENSE`.
