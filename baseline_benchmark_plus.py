
import os
import csv
import glob
import math
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
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, rdMolDescriptors

# ===== ( baseline_benchmark ,) =====
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
    candidates += glob.glob(os.path.join('data', '**', '*.smi'), recursive=True)
    candidates += glob.glob(os.path.join('data', '**', '*.sdf'), recursive=True)

    smiles = []
    for p in candidates:
        smiles += load_smiles_from_file(p, limit=None)
        if limit and len(smiles) >= limit:
            smiles = smiles[:limit]
            break
    if not smiles:
        smiles = ['c1ccccc1', 'c1ccncc1', 'CCO', 'CCN', 'CC(=O)O',
                  'c1ccc2[nH]cnc2c1', 'C1CCCCC1', 'CCc1ccc2[nH]cnc2c1']
    return smiles

def compute_sa_norm(mol):
    try:
        raw = rdMolDescriptors.CalcSAScore(mol)
    except Exception:
        try:
            from rdkit.Chem import SA_Score
            raw = SA_Score.calculateScore(mol)
        except Exception:
            rings = mol.GetRingInfo().NumRings()
            heavy = Descriptors.HeavyAtomCount(mol)
            hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
            approx = 2.0 + 0.3 * rings + 0.01 * heavy + 0.2 * hetero
            raw = float(max(1.0, min(10.0, approx)))
    sa_norm = 1.0 - (raw - 1.0) / 9.0
    return max(0.0, min(1.0, sa_norm))

def compute_synth_complexity(mol):
    rings = mol.GetRingInfo().NumRings()
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    heavy = Descriptors.HeavyAtomCount(mol)
    logp = Crippen.MolLogP(mol)
    score = 0.6 * rings + 0.4 * arom + 0.2 * (rot / 5.0) + 0.2 * (heavy / 20.0) + 0.1 * max(0.0, logp / 5.0)
    return float(score)

def safe_mol(s):
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

def evaluate_smiles_list(smiles_list):
    rows = []
    for s in smiles_list:
        m = safe_mol(s)
        if m is None:
            continue
        try:
            sa = compute_sa_norm(m)
            qed = QED.qed(m)
            sc = compute_synth_complexity(m)
            ok = (sa >= 0.7) and (qed >= 0.5) and (sc > 1.5)
            rows.append({'SMILES': s, 'SA_norm': sa, 'QED': float(qed), 'SynthComplex': sc, 'MeetsCriteria': int(ok)})
        except Exception:
            continue
    return rows

def load_smiles_from_csv(csv_path: str, smiles_col: str = None, limit: int = None):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows
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
                rows.append(s)
                if limit and len(rows) >= limit:
                    break
    return rows

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    margin = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)
    lower = (centre - margin)/denom
    upper = (centre + margin)/denom
    return max(0.0, lower), min(1.0, upper)

# ===== /() =====
def simple_mutate(s):
    choices = ['C', 'N', 'O', 'F', 'Cl']
    return s + random.choice(choices)

def gen_flag4(seeds, num):
    try:
        from rdkit.Chem import BRICS
        frags = []
        for s in seeds:
            m = safe_mol(s)
            if m is None:
                continue
            cuts = list(BRICS.BRICSDecompose(m))
            frags.extend(list(cuts))
        frags = list(set(frags))[:50]
        out = []
        while len(out) < num and frags:
            s = random.choice(seeds)
            f = random.choice(frags)
            cand = s + f
            if Chem.MolFromSmiles(cand):
                out.append(cand)
        while len(out) < num:
            out.append(simple_mutate(random.choice(seeds)))
        return out
    except Exception:
        return [simple_mutate(random.choice(seeds)) for _ in range(num)]

def gen_reinvent7(seeds, num):
    paths = glob.glob(os.path.join('data', 'results', '**', 'memory.smi'), recursive=True)
    pool = []
    for p in paths:
        pool += load_smiles_from_file(p)
    if pool:
        return [random.choice(pool) for _ in range(num)]
    return [simple_mutate(random.choice(seeds)) for _ in range(num)]

def gen_deepfmpo(seeds, num):
    out = []
    for s in seeds:
        m = safe_mol(s)
        if not m:
            continue
        try:
            mh = Chem.AddHs(m)
            AllChem.EmbedMolecule(mh, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mh)
            rings = m.GetRingInfo().NumRings()
            arom = rdMolDescriptors.CalcNumAromaticRings(m)
            if rings + arom >= 2:
                out.append(Chem.MolToSmiles(m))
        except Exception:
            continue
        if len(out) >= num:
            break
    while len(out) < num:
        out.append(simple_mutate(random.choice(seeds)))
    return out

def load_diffgat_smiles(ours_csv: str = None):
    candidates = []
    if ours_csv and os.path.isfile(ours_csv):
        candidates.append(ours_csv)
    else:
        candidates += glob.glob('molecules_report.csv')
        candidates += glob.glob('*generated_molecules*.csv')
        candidates += glob.glob('*ours*.csv')
        candidates.append(r'C:\\Users\\liuzy\\Desktop\\molecules_report.csv')
        home = os.path.expanduser('~')
        candidates.append(os.path.join(home, 'Desktop', 'molecules_report.csv'))
        candidates.append(os.path.join(home, '', 'molecules_report.csv'))
    for p in candidates:
        if os.path.isfile(p):
            rows = []
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        s = row.get('SMILES') or row.get('smiles')
                        if s:
                            rows.append(s)
            except Exception:
                pass
            if rows:
                print(f" DiffGAT CSV:{p} (n={len(rows)})")
                return rows
    print(' DiffGAT CSV')
    return []

# =====  =====
def plot_bar_with_ci(summary, outfile):
    if plt is None:
        return
    labels = [r['Model'] for r in summary]
    ratios = [r['HitRatio'] for r in summary]
    lowers = [r['CI_low'] for r in summary]
    uppers = [r['CI_high'] for r in summary]
    errs = [
        [max(0.0, ratios[i] - lowers[i]) for i in range(len(ratios))],
        [max(0.0, uppers[i] - ratios[i]) for i in range(len(ratios))]
    ]
    plt.figure(figsize=(7,4))
    colors = sns.color_palette('deep', n_colors=len(labels)) if sns else None
    if colors is None:
        plt.bar(labels, ratios, yerr=errs, capsize=4)
    else:
        plt.bar(labels, ratios, yerr=errs, color=colors, capsize=4)
    plt.axhline(0.5, ls='--', c='gray', lw=1)
    for i,(x,y) in enumerate(zip(labels, ratios)):
        plt.text(i, y+0.02, f"n={summary[i]['Count']}", ha='center', fontsize=9)
    plt.ylabel('Proportion Meeting Criteria')
    plt.ylim(0,1.0)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

def plot_resample_violin(distributions: dict, outfile: str):
    if plt is None:
        return
    data = []
    labels = []
    for k,v in distributions.items():
        labels.append(k)
        data.append(v)
    plt.figure(figsize=(7,4))
    sns.violinplot(data=data, inner='box') if sns else plt.boxplot(data)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel('HitRatio (Resampled, size = n_DiffGAT)')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

# =====  =====
def run_benchmark_plus(casf_dir: str, num_samples: int, ours_csv: str = None, resample_runs: int = 1000,
                       allow_proxy_baselines: bool = False, baseline_csv_pairs=None):
    models = {}
    if baseline_csv_pairs:
        for model_name, csv_path in baseline_csv_pairs:
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"Baseline CSV not found: {csv_path}")
            models[model_name] = load_smiles_from_csv(csv_path, limit=num_samples)
    elif allow_proxy_baselines:
        seeds = load_casf2016_core_dataset(casf_dir, limit=max(100, num_samples // 2))
        random.shuffle(seeds)
        models = {
            'FLAG4(proxy)': gen_flag4(seeds, num_samples),
            'REINVENT7(proxy)': gen_reinvent7(seeds, num_samples),
            'DeepFMPO1(proxy)': gen_deepfmpo(seeds, num_samples)
        }
    else:
        raise RuntimeError(
            "Strict mode requires --baseline_csv MODEL CSV pairs. "
            "Use --allow_proxy_baselines only for demo/debug."
        )

    if not ours_csv or (not os.path.isfile(ours_csv)):
        raise FileNotFoundError("DiffGAT CSV is required. Please provide --ours_csv.")
    models['DiffGAT'] = load_smiles_from_csv(ours_csv, limit=num_samples)

    # 
    eval_results = {name: evaluate_smiles_list(smi_list) for name, smi_list in models.items()}
    summary = []
    for name, rows in eval_results.items():
        n = len(rows)
        k = sum(r['MeetsCriteria'] for r in rows)
        ratio = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        summary.append({'Model': name, 'Count': n, 'HitRatio': ratio, 'CI_low': lo, 'CI_high': hi})

    os.makedirs('figures', exist_ok=True)
    plot_bar_with_ci(summary, os.path.join('figures', 'baseline_bar_ci.png'))

    # ( DiffGAT )
    if 'DiffGAT' in models:
        target_n = len(models['DiffGAT'])
        if target_n >= 2:
            distributions = {}
            #  DiffGAT ,(0)
            diffgat_rows = evaluate_smiles_list(models['DiffGAT'])
            dn = len(diffgat_rows)
            dk = sum(r['MeetsCriteria'] for r in diffgat_rows)
            d_ratio = dk / dn if dn else 0.0
            distributions['DiffGAT'] = [d_ratio] * resample_runs

            for name, smi_list in models.items():
                if name == 'DiffGAT':
                    continue
                if len(smi_list) < target_n:
                    continue
                ratios = []
                for _ in range(resample_runs):
                    sample = random.sample(smi_list, target_n)
                    rows = evaluate_smiles_list(sample)
                    n = len(rows)
                    k = sum(r['MeetsCriteria'] for r in rows)
                    ratios.append(k / n if n else 0.0)
                distributions[name] = ratios
            # ,
            ordered = {}
            for key in models.keys():
                if key in distributions:
                    ordered[key] = distributions[key]
            plot_resample_violin(ordered, os.path.join('figures', 'baseline_resample_violin.png'))

    # CSV()
    with open('results_baseline_summary_plus.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model','Count','HitRatio','CI_low','CI_high'])
        writer.writeheader()
        for r in summary:
            writer.writerow(r)

    print(':figures/baseline_bar_ci.png (DiffGAT)figures/baseline_resample_violin.png')
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--casf_dir', type=str, default=None)
    parser.add_argument('--num', type=int, default=500)
    parser.add_argument('--ours_csv', type=str, default=None)
    parser.add_argument('--resample_runs', type=int, default=1000)
    parser.add_argument('--baseline_csv', nargs=2, action='append', metavar=('MODEL', 'CSV'),
                        help='External baseline output CSV pair; repeatable.')
    parser.add_argument('--allow_proxy_baselines', action='store_true',
                        help='允许运行脚本内置代理基线（仅演示，不建议用于论文主对比）')
    args = parser.parse_args()
    run_benchmark_plus(
        args.casf_dir, args.num, args.ours_csv, args.resample_runs, args.allow_proxy_baselines, args.baseline_csv
    )

if __name__ == '__main__':
    main()