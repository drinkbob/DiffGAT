#!/usr/bin/env python3
"""
Reproducibility preflight checker for DiffGAT_paper_only.

What it checks:
1) Required python dependencies import correctly.
2) Required project files exist.
3) Key scripts pass syntax compile.
4) Optional: baseline CSV inputs contain a SMILES column.
5) Optional: training CSV for ablation novelty contains a SMILES column.

Usage examples:
  python repro_check.py
  python repro_check.py --train_csv benzimidazole_dataset.csv
  python repro_check.py --baseline_csv REINVENT7 reinvent_out.csv --baseline_csv DeepFMPO1 deepfmpo_out.csv
"""

import os
import json
import argparse
import importlib
import py_compile
import csv
from datetime import datetime


DEPENDENCIES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "rdkit",
    "torch",
    "torch_geometric",
]

REQUIRED_FILES = [
    "improved_hybrid_molecular_generator.py",
    "baseline_benchmark.py",
    "baseline_benchmark_plus.py",
    "ablation_evaluation.py",
    "compute_core_retention.py",
    "optimized_demo_ablation.py",
    "optimized_demo_ablation_synthetic.py",
    "requirements.txt",
]

COMPILE_TARGETS = [
    "improved_hybrid_molecular_generator.py",
    "baseline_benchmark.py",
    "baseline_benchmark_plus.py",
    "ablation_evaluation.py",
    "compute_core_retention.py",
    "optimized_demo_ablation.py",
    "optimized_demo_ablation_synthetic.py",
]


def check_imports():
    out = []
    for mod in DEPENDENCIES:
        try:
            importlib.import_module(mod)
            out.append({"module": mod, "ok": True, "error": ""})
        except Exception as e:
            out.append({"module": mod, "ok": False, "error": str(e)})
    return out


def check_files_exist():
    out = []
    for path in REQUIRED_FILES:
        out.append({"file": path, "ok": os.path.isfile(path)})
    return out


def check_compile():
    out = []
    for path in COMPILE_TARGETS:
        try:
            py_compile.compile(path, doraise=True)
            out.append({"file": path, "ok": True, "error": ""})
        except Exception as e:
            out.append({"file": path, "ok": False, "error": str(e)})
    return out


def detect_smiles_column(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        for cand in ("SMILES", "smiles", "Smiles"):
            if cand in reader.fieldnames:
                return cand
        return None


def check_csv_with_smiles(csv_path: str):
    if not os.path.isfile(csv_path):
        return {"csv": csv_path, "ok": False, "error": "file not found", "smiles_col": None}
    try:
        col = detect_smiles_column(csv_path)
        if col is None:
            return {"csv": csv_path, "ok": False, "error": "SMILES column missing", "smiles_col": None}
        return {"csv": csv_path, "ok": True, "error": "", "smiles_col": col}
    except Exception as e:
        return {"csv": csv_path, "ok": False, "error": str(e), "smiles_col": None}


def summarize(report):
    total = 0
    failed = 0
    for section in ("imports", "files", "compile", "baseline_csv", "train_csv"):
        for row in report.get(section, []):
            total += 1
            if not row.get("ok", False):
                failed += 1
    return total, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default=None, help="Training CSV used by ablation_evaluation novelty.")
    parser.add_argument("--baseline_csv", nargs=2, action="append", metavar=("MODEL", "CSV"),
                        help="External baseline CSV pair; repeatable.")
    parser.add_argument("--out_json", type=str, default="repro_check_report.json")
    args = parser.parse_args()

    report = {
        "generated_at": datetime.now().isoformat(),
        "imports": check_imports(),
        "files": check_files_exist(),
        "compile": check_compile(),
        "baseline_csv": [],
        "train_csv": [],
    }

    if args.baseline_csv:
        for model_name, path in args.baseline_csv:
            row = check_csv_with_smiles(path)
            row["model"] = model_name
            report["baseline_csv"].append(row)

    if args.train_csv:
        report["train_csv"].append(check_csv_with_smiles(args.train_csv))

    total, failed = summarize(report)
    report["summary"] = {
        "total_checks": total,
        "failed_checks": failed,
        "passed": failed == 0,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Repro check report written: {args.out_json}")
    print(f"[INFO] total_checks={total}, failed_checks={failed}")
    if failed > 0:
        print("[WARN] Reproducibility preflight failed. See report for details.")
    else:
        print("[OK] Reproducibility preflight passed.")


if __name__ == "__main__":
    main()
