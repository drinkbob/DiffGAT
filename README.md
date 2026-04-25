# DiffGAT Code Package

This repository contains a compact implementation of a molecular generation workflow, including:

- model training and molecule generation,
- ablation evaluation,
- baseline comparison,
- scaffold/core retention analysis,
- reproducibility checks.

The focus of this README is practical usage only.

---

## 1. Project Structure

Main scripts:

- `improved_hybrid_molecular_generator.py`  
  Main training and generation pipeline.

- `multi_core_molecular_generator.py`  
  Multi-core constrained generation (not limited to benzimidazole).

- `ablation_evaluation.py`  
  Ablation runs and metric/plot export.

- `baseline_benchmark.py`  
  Baseline comparison in strict external-CSV mode.

- `baseline_benchmark_plus.py`  
  Extended baseline comparison (confidence intervals and resampling).

- `compute_core_retention.py`  
  Core scaffold retention statistics.

- `repro_check.py`  
  Environment/input preflight checker.

- `requirements.txt`  
  Python dependency list.

---

## 2. Requirements

- Python 3.9+ (3.9.x recommended)
- RDKit-compatible environment

Install dependencies:

```bash
py -3 -m pip install -r requirements.txt
```

> Note: `requirements.txt` uses `numpy<2` for RDKit compatibility.

---

## 3. Input Data Format

Any input CSV used for training/evaluation should contain one SMILES column:

- `SMILES`, or
- `smiles`, or
- `Smiles`

Example:

```csv
SMILES
c1ccc2[nH]cnc2c1
CCc1ccc2[nH]cnc2c1
```

---

## 4. Quick Start

### Step 0: Run preflight checks

```bash
py -3 repro_check.py --train_csv benzimidazole_dataset.csv
```

This verifies dependencies, key files, and script compile status.

### Step 1: Train and generate molecules

```bash
py -3 improved_hybrid_molecular_generator.py train 50 benzimidazole_dataset.csv
```

Typical outputs:

- `generated_molecules_YYYYMMDD_HHMMSS.csv`
- `generated_molecules_YYYYMMDD_HHMMSS.sdf`
- `best_model.pth`

### Step 2: Run ablation evaluation

```bash
py -3 ablation_evaluation.py 100 medium --train_csv benzimidazole_dataset.csv
```

Typical output directory:

- `ablation_results_YYYYMMDD_HHMMSS/`

### Step 3: Run baseline comparison (strict mode)

You must provide external baseline result CSVs.

```bash
py -3 baseline_benchmark.py --num 500 \
  --ours_csv diffgat_generated.csv \
  --baseline_csv REINVENT7 reinvent7_generated.csv \
  --baseline_csv DeepFMPO1 deepfmpo1_generated.csv
```

Extended version:

```bash
py -3 baseline_benchmark_plus.py --num 500 --resample_runs 1000 \
  --ours_csv diffgat_generated.csv \
  --baseline_csv REINVENT7 reinvent7_generated.csv \
  --baseline_csv DeepFMPO1 deepfmpo1_generated.csv
```

### Step 4: Compute core retention

```bash
py -3 compute_core_retention.py \
  --csv diffgat_generated.csv --model DiffGAT \
  --csv_baseline reinvent7_generated.csv REINVENT7 \
  --csv_baseline deepfmpo1_generated.csv DeepFMPO1 \
  --out_prefix core_retention
```

---

## 5. Multi-Core Generation

`multi_core_molecular_generator.py` supports both automatic core discovery and manual core input.

### Auto-detect top cores from dataset

```bash
py -3 multi_core_molecular_generator.py \
  --input_csv benzimidazole_dataset.csv \
  --top_k_cores 5 \
  --per_core 30 \
  --complexity medium \
  --seed 42
```

### Manually specify cores

```bash
py -3 multi_core_molecular_generator.py \
  --core_smiles "c1ccc2[nH]cnc2c1" \
  --core_smiles "c1ccc2c(c1)[nH]cc2" \
  --core_smiles "c1ccc2ncccc2c1" \
  --per_core 20 \
  --complexity medium \
  --seed 42
```

---

## 6. Reproducibility Controls

`multi_core_molecular_generator.py` supports end-to-end seed control:

- Python `random`
- `numpy`
- `torch` / CUDA (when available)
- deterministic torch backend settings

Default:

```bash
--seed 42
```

Optional (not recommended for reproducibility):

```bash
--non_deterministic_torch
```

---

## 7. Metrics Summary

Common metrics used in scripts:

- **Validity**: fraction of RDKit-parseable generated molecules
- **Uniqueness**: fraction of unique SMILES
- **Core Retention**: fraction containing the required core scaffold
- **Novelty** (in ablation): average max similarity to training set molecules
- **HitRatio** (benchmark): fraction passing SA/QED/complexity thresholds

---

## 8. Troubleshooting

### RDKit import or binary errors

- Reinstall dependencies from `requirements.txt`
- Ensure `numpy<2` is active in your environment

### Baseline benchmark raises strict-mode error

- Provide `--baseline_csv MODEL_NAME PATH.csv` pairs
- Provide `--ours_csv` for DiffGAT outputs

### No SMILES column found

- Rename your column to one of:
  - `SMILES`
  - `smiles`
  - `Smiles`

---

## 9. Notes

- Keep generated outputs outside version control if you want a code-only workspace.
- Use `repro_check.py` before running long experiments.
