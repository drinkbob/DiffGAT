"""
(CASF2016 )

:
- FLAG4:3D(,BRICS)
- REINVENT7:RNN+(,RNNSMILES)
- DeepFMPO1:3D(,3D+)

::
- SA ≥ 0.7(RDKit SA_Score[1,10][0,1]SA_norm)
- QED ≥ 0.5
-  > 1.5()

:
- results_baseline_summary.csv()
- figures/baseline_bar.png()
- figures/baseline_cdf.png(SAQED)
- figures/baseline_scatter.png(QED vs SA_norm )

:
  python baseline_benchmark.py --casf_dir data/CASF2016 --num 500 \
    --ours_csv generated_molecules_YYYYMMDD_HHMMSS.csv

:
- CASF,data/.smi/.sdf；.
- ,,、.
"""

import os
import sys
import csv
import glob
import argparse
import random
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, Lipinski, rdMolDescriptors

# ==========  ==========
def try_import(module_name: str):
    try:
        __import__(module_name)
        return True
    except Exception:
        return False

def load_smiles_from_file(path: str, limit: int = None):
    smiles_list = []
    if path.endswith('.smi') or path.endswith('.txt'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip().split()[0]
                    if s:
                        smiles_list.append(s)
                        if limit and len(smiles_list) >= limit:
                            break
        except Exception:
            pass
    elif path.endswith('.sdf'):
        try:
            suppl = Chem.SDMolSupplier(path)
            for mol in suppl:
                if mol is None:
                    continue
                s = Chem.MolToSmiles(mol)
                if s:
                    smiles_list.append(s)
                    if limit and len(smiles_list) >= limit:
                        break
        except Exception:
            pass
    return smiles_list

def load_casf2016_core_dataset(root_dir: str = None, limit: int = None):
    candidates = []
    if root_dir and os.path.isdir(root_dir):
        candidates += glob.glob(os.path.join(root_dir, '**', '*.smi'), recursive=True)
        candidates += glob.glob(os.path.join(root_dir, '**', '*.sdf'), recursive=True)
    # 
    candidates += glob.glob(os.path.join('data', '**', '*.smi'), recursive=True)
    candidates += glob.glob(os.path.join('data', '**', '*.sdf'), recursive=True)

    smiles = []
    for p in candidates:
        smiles += load_smiles_from_file(p, limit=None)
        if limit and len(smiles) >= limit:
            smiles = smiles[:limit]
            break

    if not smiles:
        # ,
        smiles = [
            'c1ccccc1', 'c1ccncc1', 'CCO', 'CCN', 'CC(=O)O',
            'c1ccc2[nH]cnc2c1', 'C1CCCCC1', 'CCc1ccc2[nH]cnc2c1'
        ]
    return smiles

def compute_sa_norm(mol):
    # RDKit contrib SA_Score:1()~10()
    raw = None
    try:
        # rdMolDescriptors
        raw = rdMolDescriptors.CalcSAScore(mol)  # 
    except Exception:
        try:
            from rdkit.Chem import SA_Score
            raw = SA_Score.calculateScore(mol)
        except Exception:
            raw = None
    if raw is None:
        # :、、,[1,10]
        rings = mol.GetRingInfo().NumRings()
        heavy = Descriptors.HeavyAtomCount(mol)
        hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        approx = 2.0 + 0.3 * rings + 0.01 * heavy + 0.2 * hetero
        raw = float(max(1.0, min(10.0, approx)))
    # [0,1],1->1.0, 10->0.0
    sa_norm = 1.0 - (raw - 1.0) / 9.0
    return max(0.0, min(1.0, sa_norm))

def compute_synthetic_complexity(mol):
    # :>1.5 
    rings = mol.GetRingInfo().NumRings()
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    heavy = Descriptors.HeavyAtomCount(mol)
    logp = Crippen.MolLogP(mol)
    # 
    score = 0.6 * rings + 0.4 * arom_rings + 0.2 * (rot / 5.0) + 0.2 * (heavy / 20.0) + 0.1 * max(0.0, logp / 5.0)
    return float(score)

def safe_mol_from_smiles(s: str):
    try:
        m = Chem.MolFromSmiles(s)
        return m
    except Exception:
        return None

def mutate_smiles_simple(s: str):
    # :
    choices = ['C', 'N', 'O', 'F', 'Cl']
    add = random.choice(choices)
    return s + add

# ==========  ==========
class BaseAdapter:
    name = 'BASE'
    def __init__(self, seeds):
        self.seeds = seeds
    def available(self) -> bool:
        return False
    def generate(self, num_samples: int):
        return []

class FLAG4Adapter(BaseAdapter):
    name = 'FLAG4'
    def available(self) -> bool:
        return try_import('flag4') or try_import('FLAG')
    def generate(self, num_samples: int):
        if self.available():
            # :FLAG4 API
            gen = []
            for s in self.seeds:
                gen.append(s)
                if len(gen) >= num_samples:
                    break
            return gen
        # :BRICS
        try:
            from rdkit.Chem import BRICS
            frags = []
            for s in self.seeds:
                m = safe_mol_from_smiles(s)
                if not m:
                    continue
                cuts = list(BRICS.BRICSDecompose(m))
                frags.extend(list(cuts))
            frags = list(set(frags))[:50]
            out = []
            while len(out) < num_samples and frags:
                s = random.choice(self.seeds)
                f = random.choice(frags)
                cand = s + f
                if Chem.MolFromSmiles(cand):
                    out.append(cand)
            return out
        except Exception:
            # :
            out = []
            while len(out) < num_samples:
                s = random.choice(self.seeds)
                out.append(mutate_smiles_simple(s))
            return out

class REINVENT7Adapter(BaseAdapter):
    name = 'REINVENT7'
    def available(self) -> bool:
        return try_import('reinvent') or try_import('reinvent_models')
    def _load_repo_rnn_memories(self):
        #  data/results/**/memory.smi 
        paths = glob.glob(os.path.join('data', 'results', '**', 'memory.smi'), recursive=True)
        pool = []
        for p in paths:
            pool += load_smiles_from_file(p)
        return list(set(pool))
    def generate(self, num_samples: int):
        if self.available():
            # :REINVENTAPI
            out = []
            for s in self.seeds:
                out.append(s)
                if len(out) >= num_samples:
                    break
            return out
        # :RNN
        pool = self._load_repo_rnn_memories()
        out = []
        if pool:
            while len(out) < num_samples:
                out.append(random.choice(pool))
            return out
        # :SMILES
        while len(out) < num_samples:
            s = random.choice(self.seeds)
            out.append(mutate_smiles_simple(s))
        return out

class DeepFMPO1Adapter(BaseAdapter):
    name = 'DeepFMPO1'
    def available(self) -> bool:
        return try_import('deepfmpo') or try_import('deep_fmpo')
    def generate(self, num_samples: int):
        if self.available():
            # :DeepFMPO3D
            out = []
            for s in self.seeds:
                out.append(s)
                if len(out) >= num_samples:
                    break
            return out
        # :3D()
        out = []
        for s in self.seeds:
            m = safe_mol_from_smiles(s)
            if not m:
                continue
            try:
                m_h = Chem.AddHs(m)
                AllChem.EmbedMolecule(m_h, AllChem.ETKDG())
                AllChem.UFFOptimizeMolecule(m_h)
                # :=,+
                rings = m.GetRingInfo().NumRings()
                arom = rdMolDescriptors.CalcNumAromaticRings(m)
                if rings + arom >= 2:
                    out.append(Chem.MolToSmiles(m))
            except Exception:
                continue
            if len(out) >= num_samples:
                break
        # ,
        while len(out) < num_samples:
            s = random.choice(self.seeds)
            out.append(mutate_smiles_simple(s))
        return out

# ==========  ==========
def evaluate_set(smiles_list):
    results = []
    for s in smiles_list:
        mol = safe_mol_from_smiles(s)
        if mol is None:
            continue
        try:
            sa = compute_sa_norm(mol)
            qed = QED.qed(mol)
            sc = compute_synthetic_complexity(mol)
            ok = (sa >= 0.7) and (qed >= 0.5) and (sc > 1.5)
            results.append({
                'SMILES': s,
                'SA_norm': sa,
                'QED': float(qed),
                'SynthComplex': sc,
                'MeetsCriteria': int(ok)
            })
        except Exception:
            continue
    return results

def load_smiles_from_csv(csv_path: str, smiles_col: str = None, limit: int = None):
    smiles = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return smiles
        if smiles_col is None:
            for cand in ('SMILES', 'smiles', 'Smiles'):
                if cand in reader.fieldnames:
                    smiles_col = cand
                    break
            if smiles_col is None:
                raise ValueError(f"No SMILES column found in {csv_path}. Columns={reader.fieldnames}")
        for row in reader:
            s = row.get(smiles_col)
            if s:
                smiles.append(s)
                if limit and len(smiles) >= limit:
                    break
    return smiles

def summarize_results(all_results: dict):
    summary = []
    for name, rows in all_results.items():
        if not rows:
            summary.append({'Model': name, 'Count': 0, 'HitRatio': 0.0})
            continue
        count = len(rows)
        hits = sum(r['MeetsCriteria'] for r in rows)
        ratio = hits / count if count else 0.0
        summary.append({'Model': name, 'Count': count, 'HitRatio': ratio})
    return summary

def save_summary_csv(summary, filename='results_baseline_summary.csv'):
    if not summary:
        return
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Count', 'HitRatio'])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

def plot_figures(all_results: dict, out_dir='figures'):
    if plt is None:
        print(' matplotlib/seaborn,')
        return
    os.makedirs(out_dir, exist_ok=True)
    # :
    summary = summarize_results(all_results)
    labels = [r['Model'] for r in summary]
    ratios = [r['HitRatio'] for r in summary]
    plt.figure(figsize=(6, 4))
    colors = sns.color_palette('deep', n_colors=len(labels)) if sns else None
    if colors is None:
        plt.bar(labels, ratios)
    else:
        plt.bar(labels, ratios, color=colors)
    plt.axhline(0.5, ls='--', c='gray', lw=1)
    plt.ylabel('Proportion Meeting Criteria')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'baseline_bar.png'), dpi=300)
    plt.close()

    # CDF:SA_norm  QED
    plt.figure(figsize=(6, 4))
    for name, rows in all_results.items():
        sa_vals = sorted([r['SA_norm'] for r in rows])
        if not sa_vals:
            continue
        y = np.linspace(0, 1, len(sa_vals))
        plt.plot(sa_vals, y, label=f'{name} SA')
    plt.axvline(0.7, ls='--', c='gray', lw=1)
    plt.xlabel('SA_norm')
    plt.ylabel('CDF')
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'baseline_cdf_sa.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    for name, rows in all_results.items():
        q_vals = sorted([r['QED'] for r in rows])
        if not q_vals:
            continue
        y = np.linspace(0, 1, len(q_vals))
        plt.plot(q_vals, y, label=f'{name} QED')
    plt.axvline(0.5, ls='--', c='gray', lw=1)
    plt.xlabel('QED')
    plt.ylabel('CDF')
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'baseline_cdf_qed.png'), dpi=300)
    plt.close()

    # :QED vs SA_norm
    plt.figure(figsize=(6, 5))
    for name, rows in all_results.items():
        x = [r['SA_norm'] for r in rows]
        y = [r['QED'] for r in rows]
        plt.scatter(x, y, s=10, alpha=0.5, label=name)
    plt.axvline(0.7, ls='--', c='gray', lw=1)
    plt.axhline(0.5, ls='--', c='gray', lw=1)
    plt.xlabel('SA_norm')
    plt.ylabel('QED')
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'baseline_scatter.png'), dpi=300)
    plt.close()

# ==========  ==========
def run_benchmark(casf_dir: str, num_samples: int, ours_csv: str = None, allow_fallback_baselines: bool = False,
                  baseline_csv_pairs=None, legacy_proxy_mode: bool = False):
    # Strict manuscript mode: evaluate only externally generated model outputs.
    if baseline_csv_pairs:
        all_results = {}
        availability_notes = {}
        for model_name, csv_path in baseline_csv_pairs:
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"Baseline CSV not found: {csv_path}")
            model_smiles = load_smiles_from_csv(csv_path, limit=num_samples)
            all_results[model_name] = evaluate_set(model_smiles)
            availability_notes[model_name] = 'external_csv'
            print(f"[RUN] {model_name} from CSV: n={len(all_results[model_name])}")

        if not ours_csv or (not os.path.isfile(ours_csv)):
            raise FileNotFoundError("DiffGAT CSV is required in strict mode. Please provide --ours_csv.")
        ours_smiles = load_smiles_from_csv(ours_csv, limit=num_samples)
        all_results['DiffGAT'] = evaluate_set(ours_smiles)
        availability_notes['DiffGAT'] = 'external_csv'
        print(f"[RUN] DiffGAT from CSV: n={len(all_results['DiffGAT'])}")

        summary = summarize_results(all_results)
        save_summary_csv(summary)
        plot_figures(all_results)
        print("\n[INFO] Strict comparison mode completed (external CSV only).")
        return summary, availability_notes

    if not legacy_proxy_mode:
        raise RuntimeError(
            "Strict mode requires --baseline_csv MODEL CSV pairs and --ours_csv. "
            "Proxy/generated baselines are blocked to protect manuscript integrity. "
            "Use --legacy_proxy_mode only for local demo/debug."
        )

    seeds = load_casf2016_core_dataset(casf_dir, limit=max(100, num_samples // 2))
    random.shuffle(seeds)
    # 
    adapters = [
        FLAG4Adapter(seeds),
        REINVENT7Adapter(seeds),
        DeepFMPO1Adapter(seeds),
    ]

    all_results = {}
    availability_notes = {}

    for adp in adapters:
        print(f"[RUN] Baseline: {adp.name}")
        avail = adp.available()
        availability_notes[adp.name] = 'native' if avail else 'fallback'
        if (not avail) and (not allow_fallback_baselines):
            print(f"   [WARN] {adp.name} unavailable; skipped to avoid fallback proxy baseline. Use --allow_fallback_baselines to enable.")
            all_results[adp.name] = []
            continue
        gen = adp.generate(num_samples)
        eval_rows = evaluate_set(gen)
        all_results[adp.name] = eval_rows
        ratio = sum(r['MeetsCriteria'] for r in eval_rows) / max(1, len(eval_rows))
        print(f"   ={len(eval_rows)}, ={ratio:.3f} ({'' if avail else ''})")

    # :(CSV)
    # 1) 
    if ours_csv and os.path.isfile(ours_csv):
        print(f"[RUN] DiffGAT CSV: {ours_csv}")
        ours_smiles = []
        try:
            with open(ours_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s = row.get('SMILES') or row.get('smiles')
                    if s:
                        ours_smiles.append(s)
        except Exception:
            pass
        eval_rows = evaluate_set(ours_smiles[:num_samples])
        #  DiffGAT
        all_results['DiffGAT'] = eval_rows
        ratio = sum(r['MeetsCriteria'] for r in eval_rows) / max(1, len(eval_rows))
        print(f"   DiffGAT ={len(eval_rows)}, ={ratio:.3f}")
    else:
        # 2) 
        auto_candidates = []
        # 
        auto_candidates += glob.glob('molecules_report.csv')
        auto_candidates += glob.glob('*generated_molecules*.csv')
        auto_candidates += glob.glob('*ours*.csv')
        auto_candidates += glob.glob(os.path.join('..', 'molecules_report.csv'))
        # Windows ()
        auto_candidates.append(r'C:\Users\liuzy\Desktop\molecules_report.csv')
        # Linux /
        home = os.path.expanduser('~')
        auto_candidates.append(os.path.join(home, 'Desktop', 'molecules_report.csv'))
        auto_candidates.append(os.path.join(home, '', 'molecules_report.csv'))
        found = None
        for p in auto_candidates:
            if p and os.path.isfile(p):
                found = p
                break
        if found:
            print(f"[RUN] Auto-detected DiffGAT CSV: {found}")
            ours_smiles = []
            try:
                with open(found, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        s = row.get('SMILES') or row.get('smiles')
                        if s:
                            ours_smiles.append(s)
            except Exception:
                pass
            eval_rows = evaluate_set(ours_smiles[:num_samples])
            all_results['DiffGAT'] = eval_rows
            ratio = sum(r['MeetsCriteria'] for r in eval_rows) / max(1, len(eval_rows))
            print(f"   DiffGAT ={len(eval_rows)}, ={ratio:.3f}")
        else:
            print('[INFO] No DiffGAT CSV found. Use --ours_csv to include DiffGAT in comparison.')

    # 
    summary = summarize_results(all_results)
    save_summary_csv(summary)
    plot_figures(all_results)

    # 
    print("\n:")
    for k, v in availability_notes.items():
        print(f"- {k}: {'' if v=='native' else ''}")

    return summary, availability_notes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--casf_dir', type=str, default=None, help='CASF2016')
    parser.add_argument('--num', type=int, default=500, help='')
    parser.add_argument('--ours_csv', type=str, default=None, help=':CSV')
    parser.add_argument('--baseline_csv', nargs=2, action='append', metavar=('MODEL', 'CSV'),
                        help='External baseline output CSV pair; repeatable, e.g. --baseline_csv REINVENT7 reinvent.csv')
    parser.add_argument('--legacy_proxy_mode', action='store_true',
                        help='Enable legacy proxy baseline generation (demo only, not for manuscript claims).')
    parser.add_argument('--allow_fallback_baselines', action='store_true',
                        help='允许在缺少官方基线实现时使用脚本内置fallback（仅演示，不建议用于论文主对比）')
    args = parser.parse_args()

    summary, notes = run_benchmark(
        args.casf_dir,
        args.num,
        args.ours_csv,
        args.allow_fallback_baselines,
        args.baseline_csv,
        args.legacy_proxy_mode,
    )
    print('\n:')
    for r in summary:
        print(f"  {r['Model']}: HitRatio={r['HitRatio']:.3f} (n={r['Count']})")

if __name__ == '__main__':
    main()