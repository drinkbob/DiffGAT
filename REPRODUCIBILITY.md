# REPRODUCIBILITY (Appendix Grade)

This document defines a manuscript-grade, auditable reproducibility protocol for `DiffGAT`.
It is written to align with common  reviewer expectations: transparent inputs, strict command order, explicit metric semantics, and deterministic/repeated evaluation policy.

## 1) Scope and Claim Boundary

This package contains two types of scripts:

- **Manuscript-claim scripts (evidence-grade)**:
  - `improved_hybrid_molecular_generator.py`
  - `ablation_evaluation.py`
  - `baseline_benchmark.py` (strict external-CSV mode)
  - `baseline_benchmark_plus.py` (strict external-CSV mode)
  - `compute_core_retention.py`
  - `repro_check.py`
- **Demo-only scripts (non-evidence)**:
  - `optimized_demo_ablation.py`
  - `optimized_demo_ablation_synthetic.py`

Outputs from demo-only scripts are synthetic/illustrative and **must not** be used as manuscript evidence.

## 2) Environment Specification

### Python and OS

- Recommended: Python `3.9.x` (validated with `3.9.10`)
- OS validated: Windows 10/11

### Dependencies

Install from:

```bash
py -3 -m pip install -r requirements.txt
```

`requirements.txt` pins `numpy<2` for RDKit compatibility.

### Preflight validation (mandatory)

```bash
py -3 repro_check.py --train_csv benzimidazole_dataset.csv
```

Expected: `failed_checks=0` and generated `repro_check_report.json`.

## 3) Required Input Artifacts

### Core data

- `benzimidazole_dataset.csv`
  - Must include one of: `SMILES` / `smiles` / `Smiles`
  - Used as real training reference for novelty in ablation

### External baseline model outputs (strict mode)

For each baseline (e.g., REINVENT7, DeepFMPO1, FLAG4), provide a CSV containing a SMILES column:

- `reinvent7_generated.csv`
- `deepfmpo1_generated.csv`
- `flag4_generated.csv` (optional if not available)

### DiffGAT output CSV

- `diffgat_generated.csv`
  - Must include one of: `SMILES` / `smiles` / `Smiles`

## 4) Canonical Command Order (Manuscript Pipeline)

Run commands in the exact order below.

### Step 0: Reproducibility preflight

```bash
py -3 repro_check.py --train_csv benzimidazole_dataset.csv \
  --baseline_csv REINVENT7 reinvent7_generated.csv \
  --baseline_csv DeepFMPO1 deepfmpo1_generated.csv
```

### Step 1: Train / load DiffGAT and generate molecules

Training + generation:

```bash
py -3 improved_hybrid_molecular_generator.py train 50 benzimidazole_dataset.csv
```

The script exports timestamped files such as:
- `generated_molecules_YYYYMMDD_HHMMSS.csv`
- `generated_molecules_YYYYMMDD_HHMMSS.sdf`

Use the generated CSV as `diffgat_generated.csv` in later steps.

### Step 2: Ablation study (real training CSV novelty)

```bash
py -3 ablation_evaluation.py 100 medium --train_csv benzimidazole_dataset.csv
```

### Step 3: Baseline comparison (strict external CSV mode)

#### 3.1 Basic summary

```bash
py -3 baseline_benchmark.py --num 500 \
  --ours_csv diffgat_generated.csv \
  --baseline_csv REINVENT7 reinvent7_generated.csv \
  --baseline_csv DeepFMPO1 deepfmpo1_generated.csv
```

#### 3.2 CI + resampling comparison

```bash
py -3 baseline_benchmark_plus.py --num 500 --resample_runs 1000 \
  --ours_csv diffgat_generated.csv \
  --baseline_csv REINVENT7 reinvent7_generated.csv \
  --baseline_csv DeepFMPO1 deepfmpo1_generated.csv
```

### Step 4: Core retention audit

```bash
py -3 compute_core_retention.py \
  --csv diffgat_generated.csv --model DiffGAT \
  --csv_baseline reinvent7_generated.csv REINVENT7 \
  --csv_baseline deepfmpo1_generated.csv DeepFMPO1 \
  --out_prefix core_retention
```

## 5) Input/Output Mapping

| Stage | Script | Required Inputs | Primary Outputs |
|---|---|---|---|
| Preflight | `repro_check.py` | dependencies, key scripts, optional CSVs | `repro_check_report.json` |
| Train/Generate | `improved_hybrid_molecular_generator.py` | training CSV | `generated_molecules_*.csv`, `generated_molecules_*.sdf`, `best_model.pth` |
| Ablation | `ablation_evaluation.py` | generated molecules per variant (internal), `--train_csv` | `ablation_results_*/metrics_ablation.csv`, plots, report |
| Baseline summary | `baseline_benchmark.py` | external baseline CSVs + DiffGAT CSV | `results_baseline_summary.csv`, `figures/*.png` |
| Baseline CI | `baseline_benchmark_plus.py` | external baseline CSVs + DiffGAT CSV | `results_baseline_summary_plus.csv`, `figures/*.png` |
| Core audit | `compute_core_retention.py` | model output CSVs | `core_retention_*.csv`, `core_retention_summary.csv` |

## 6) Metric Definitions (Statistical Semantics)

### Generation validity metrics

- **Validity** = valid RDKit-parsable SMILES / total generated SMILES
- **Uniqueness** = unique SMILES / total generated SMILES
- **Core Retention** = molecules containing benzimidazole core / valid molecules

### Novelty metric (ablation)

In `ablation_evaluation.py`, novelty is computed as:

- For each generated molecule: compute maximum Tanimoto similarity to molecules in real training CSV
- **Novelty score (reported)** = mean of these maximum similarities
- Interpretation: **lower value means higher novelty**

### Baseline hit criterion (benchmark scripts)

A molecule is considered a hit when all are satisfied:

- `SA_norm >= 0.7`
- `QED >= 0.5`
- `SynthComplex > 1.5`

`HitRatio = hits / evaluated molecules`.

### Confidence interval (plus benchmark)

`baseline_benchmark_plus.py` reports Wilson 95% CI for HitRatio.

## 7) Random Seed Strategy

For manuscript-grade runs, use this policy:

1. **Primary deterministic seed**: `42`
2. **Repeat runs**: at least 5 independent seeds (`42, 52, 62, 72, 82`)
3. **Report**:
   - mean ± std across seeds
   - 95% CI where implemented
4. **Archive seed metadata** with each run:
   - seed list
   - command line
   - git/package snapshot (or `repro_check_report.json`)

Recommended shell setup before each run:

```bash
set PYTHONHASHSEED=42
```

If you run cross-platform or CUDA experiments, also record:
- Torch/CUDA versions
- deterministic backend flags used
- hardware model

## 8) Reviewer-Facing Audit Checklist

Before submission, verify all items are true:

- `repro_check_report.json` exists and `failed_checks=0`
- All manuscript figures/tables are traceable to non-demo scripts
- Baseline comparison uses strict external-CSV mode (no proxy mode)
- Ablation novelty used real training CSV (not proxy list)
- Command logs and generated timestamped outputs are archived
- Any synthetic/demo figure is explicitly labeled "demo only"

## 9) Explicit Non-Claim Notice

The following scripts are excluded from manuscript evidence:

- `optimized_demo_ablation.py`
- `optimized_demo_ablation_synthetic.py`

They are useful for layout/visual diagnostics only.
