#!/usr/bin/env python3
"""
Ablation evaluation for DiffGAT (ImprovedHybridMolecularGenerator)

Variants:
- Full: DiffGAT (no changes)
- -Ring: remove complex ring substituent generation (force very simple substituents)
- -Diversity: disable diversity control (no filtering by diversity buffer)
- -Ring & -Diversity: combine both ablations

Outputs (in ablation_results_YYYYMMDD_HHMMSS/):
- CSVs of generated SMILES for each variant
- metrics_ablation.csv: Validity, Uniqueness, Novelty, Core Retention for each variant
- Ablation_Metrics_Comparison.png (grouped bars)
- Ablation_Radar.png (radar chart)
- Ablation_tSNE.png (optional molecular space visualization)
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# SCI plot defaults
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.linewidth': 1.0,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Import model and SCI visualizer (optional for t-SNE)
from improved_hybrid_molecular_generator import ImprovedHybridMolecularGenerator, ImprovedHybridConfig
from sci_paper_visualization import SCIVisualizer

# -----------------------------
# Metric helpers
# -----------------------------

def smiles_to_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

def calc_validity(smiles_list: List[str]) -> float:
    valid = 0
    for s in smiles_list:
        if smiles_to_mol(s) is not None:
            valid += 1
    return valid / max(1, len(smiles_list))

def calc_uniqueness(smiles_list: List[str]) -> float:
    unique = len(set(smiles_list))
    return unique / max(1, len(smiles_list))

def tanimoto_sim(m1, m2) -> float:
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0

def calc_novelty(smiles_list: List[str], train_smiles: List[str]) -> float:
    # Novelty defined as average max similarity to training set (lower is better)
    # We report the raw value; interpretation stays in the paper text
    train_mols = [smiles_to_mol(s) for s in set(train_smiles)]
    train_mols = [m for m in train_mols if m is not None]
    if not train_mols:
        return 0.0
    max_sims = []
    for s in smiles_list:
        m = smiles_to_mol(s)
        if m is None:
            continue
        sims = [tanimoto_sim(m, tm) for tm in train_mols]
        max_sims.append(max(sims) if sims else 0.0)
    if not max_sims:
        return 0.0
    return float(np.mean(max_sims))

def calc_core_retention(smiles_list: List[str], core_smiles: str = "c1ccc2[nH]cnc2c1") -> float:
    core = Chem.MolFromSmarts(core_smiles)
    keep = 0
    total = 0
    for s in smiles_list:
        m = smiles_to_mol(s)
        if m is None:
            continue
        total += 1
        try:
            if m.HasSubstructMatch(core):
                keep += 1
        except:
            pass
    return keep / max(1, total)

def compute_metrics(smiles_list: List[str], train_smiles: List[str]) -> Dict[str, float]:
    return {
        'Validity': calc_validity(smiles_list),
        'Uniqueness': calc_uniqueness(smiles_list),
        'Novelty': calc_novelty(smiles_list, train_smiles),  # lower is better (interpret in text)
        'Core Retention': calc_core_retention(smiles_list)
    }

def load_training_smiles(train_csv: str) -> List[str]:
    if not train_csv or (not os.path.isfile(train_csv)):
        raise FileNotFoundError(
            f"Training CSV not found: {train_csv}. "
            "Please provide a real training CSV for novelty computation."
        )
    df = pd.read_csv(train_csv)
    smiles_col = None
    for cand in ('SMILES', 'smiles', 'Smiles'):
        if cand in df.columns:
            smiles_col = cand
            break
    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {train_csv}. Columns={list(df.columns)}")
    train_smiles = (
        df[smiles_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not train_smiles:
        raise ValueError(f"No valid training SMILES found in {train_csv}")
    return train_smiles

# -----------------------------
# Ablation variants
# -----------------------------

def make_full_model():
    cfg = ImprovedHybridConfig()
    return ImprovedHybridMolecularGenerator(cfg)

def make_minus_ring_model():
    # Monkey-patch substituent generator to minimize complexity
    m = make_full_model()
    def simple_substituent(_level, **kwargs):
        # use a minimal linear substituent so generation remains feasible
        return 'C'
    m._generate_benzimidazole_substituent = simple_substituent
    return m

def make_minus_diversity_model():
    m = make_full_model()
    # Disable diversity filtering
    class NoDiversity:
        def is_diverse_enough(self, _):
            return True
        def add_molecule(self, _):
            return
        def get_diversity_score(self):
            return 1.0
    m.diversity_controller = NoDiversity()
    return m

def make_minus_both_model():
    m = make_full_model()
    # Remove ring complexity
    def simple_substituent(_level, **kwargs):
        return 'C'
    m._generate_benzimidazole_substituent = simple_substituent
    # Disable diversity
    class NoDiversity:
        def is_diverse_enough(self, _):
            return True
        def add_molecule(self, _):
            return
        def get_diversity_score(self):
            return 1.0
    m.diversity_controller = NoDiversity()
    return m

# -----------------------------
# Main run
# -----------------------------

def run_variant(name: str, model_factory, num: int, complexity: str) -> List[str]:
    model = model_factory()
    generated = model.generate_molecules(num_molecules=num, complexity_level=complexity)
    smiles = [g['smiles'] for g in generated if isinstance(g, dict) and 'smiles' in g]
    return smiles

def grouped_bar_plot(metrics_dict: Dict[str, Dict[str, float]], save_path: str):
    models = list(metrics_dict.keys())
    metrics = ['Validity', 'Uniqueness', 'Novelty', 'Core Retention']
    x = np.arange(len(metrics))
    width = 0.18
    colors = ['#2E86AB', '#F18F01', '#A23B72', '#6C757D']

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        vals = [metrics_dict[model].get(m, 0) for m in metrics]
        plt.bar(x + i*width, vals, width, label=model, color=colors[i % len(colors)], alpha=0.85, edgecolor='black', linewidth=0.5)
        for j, v in enumerate(vals):
            plt.text(x[j] + i*width, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    plt.xticks(x + width*(len(models)-1)/2, metrics, fontsize=11)
    plt.ylabel('Score', fontsize=12)
    plt.title('Ablation Study: Metrics Comparison', fontsize=13, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)

def radar_plot(metrics_dict: Dict[str, Dict[str, float]], save_path: str):
    metrics = ['Validity', 'Uniqueness', 'Core Retention']  # exclude Novelty from fill for directionality
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors = ['#2E86AB', '#F18F01', '#A23B72', '#6C757D']

    for i, (model, data) in enumerate(metrics_dict.items()):
        vals = [data.get(m, 0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
        ax.fill(angles, vals, alpha=0.20, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_title('Ablation Study: Radar Plot (Excl. Novelty)', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)

def tsne_plot(all_smiles: List[str], labels: List[str], save_path: str):
    # Use SCIVisualizer for consistent styling
    vis = SCIVisualizer()
    fps = []
    valid_smiles = []
    for s in all_smiles:
        m = smiles_to_mol(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        fps.append(np.array(fp))
        valid_smiles.append(s)
    if not fps:
        return
    fps = np.array(fps)
    valid_labels = [labels[i] for i, s in enumerate(all_smiles) if s in valid_smiles]
    vis.create_tsne_plot(fps, valid_labels, title='Ablation Study: Molecular Space (t-SNE)', save_path=save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num', nargs='?', type=int, default=100)
    parser.add_argument('complexity', nargs='?', type=str, default='medium')
    parser.add_argument('--train_csv', type=str, default='benzimidazole_dataset.csv',
                        help='Real training CSV path used for novelty computation')
    args = parser.parse_args()

    num = args.num
    complexity = args.complexity

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'ablation_results_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    train_smiles = load_training_smiles(args.train_csv)
    print(f"[INFO] Loaded training SMILES for novelty: n={len(train_smiles)} from {args.train_csv}")

    variants = {
        'DiffGAT (Full)': make_full_model,
        '- Ring': make_minus_ring_model,
        '- Diversity': make_minus_diversity_model,
        '- Ring & - Diversity': make_minus_both_model,
    }

    results_smiles: Dict[str, List[str]] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for name, factory in variants.items():
        print(f"Running variant: {name}")
        smiles = run_variant(name, factory, num, complexity)
        results_smiles[name] = smiles
        metrics[name] = compute_metrics(smiles, train_smiles)
        # Save per-variant SMILES
        df = pd.DataFrame({'SMILES': smiles})
        df.to_csv(os.path.join(out_dir, f'smiles_{name.replace(" ", "_").replace("&", "and").replace("-", "minus")}.csv'), index=False)

    # Save metrics CSV
    metrics_rows = []
    for model, m in metrics.items():
        row = {'Model': model}
        row.update(m)
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(os.path.join(out_dir, 'metrics_ablation.csv'), index=False)

    # Plots
    grouped_bar_plot(metrics, os.path.join(out_dir, 'Ablation_Metrics_Comparison.png'))
    radar_plot(metrics, os.path.join(out_dir, 'Ablation_Radar.png'))

    # t-SNE across variants (optional)
    all_sm = []
    all_lb = []
    for model, smi in results_smiles.items():
        all_sm.extend(smi)
        all_lb.extend([model] * len(smi))
    if all_sm:
        tsne_plot(all_sm, all_lb, os.path.join(out_dir, 'Ablation_tSNE.png'))

    # Brief report
    with open(os.path.join(out_dir, 'ablation_report.txt'), 'w', encoding='utf-8') as f:
        f.write('Ablation Study Report\n')
        f.write('='*60 + '\n\n')
        f.write(f'Number of generated molecules per variant: {num}\n')
        f.write(f'Complexity level: {complexity}\n\n')
        f.write('Metrics (Note: Novelty is average max similarity to real training set; lower indicates higher novelty)\n')
        for model, m in metrics.items():
            f.write(f"- {model}: Validity={m['Validity']:.3f}, Uniqueness={m['Uniqueness']:.3f}, Novelty={m['Novelty']:.3f}, Core Retention={m['Core Retention']:.3f}\n")
        f.write(f"\nTraining CSV used for novelty: {args.train_csv}\n")

    print(f" Ablation evaluation complete. Results in: {out_dir}")

if __name__ == '__main__':
    main()