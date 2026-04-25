#!/usr/bin/env python3
"""
Multi-core constrained molecular generator.

This script is a generalized alternative to the benzimidazole-specific workflow in
`improved_hybrid_molecular_generator.py`. It supports:
1) automatic core recognition from a CSV SMILES dataset (Murcko scaffold frequency),
2) user-provided core list, and
3) core-preserving generation for multiple distinct cores.
"""

import os
import csv
import argparse
import random
from datetime import datetime
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


DEFAULT_CORES = [
    "c1ccc2[nH]cnc2c1",  # benzimidazole
    "c1ccc2c(c1)[nH]cc2",  # indole
    "c1ccc2ncccc2c1",  # quinoline
    "c1ccncc1",  # pyridine
    "c1ccccc1",  # benzene
]

SUBSTITUENT_BANK = {
    "simple": ["C", "CC", "CCC", "O", "CO", "N", "CN", "F", "Cl", "Br"],
    "medium": [
        "CCO", "CCN", "CC(=O)O", "OC", "NC", "SC", "S(=O)C", "C1CCCCC1",
        "c1ccccc1", "c1ccncc1", "CC(C)C", "CCOC", "CCNC"
    ],
    "complex": [
        "c1ccc(cc1)C", "c1ccc(cc1)O", "c1ccc(cc1)N", "c1ccc(cc1)Cl",
        "c1ccc2ccccc2c1", "c1ccc2c(c1)sc3ccccc23", "c1ccc2c(c1)oc3ccccc23",
        "P(=O)(O)O", "CP(=O)(O)O", "S(=O)(=O)C", "CCN(CC)CC"
    ],
}


def set_global_seed(seed: int, deterministic_torch: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch may be unavailable in lightweight environments.
        pass


def load_smiles_from_csv(csv_path, smiles_col=None, limit=None):
    smiles = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return smiles
        if smiles_col is None:
            for cand in ("SMILES", "smiles", "Smiles"):
                if cand in reader.fieldnames:
                    smiles_col = cand
                    break
            if smiles_col is None:
                raise ValueError(f"No SMILES column found in {csv_path}. Columns={reader.fieldnames}")
        for row in reader:
            s = row.get(smiles_col)
            if s:
                smiles.append(s.strip())
                if limit and len(smiles) >= limit:
                    break
    return smiles


def detect_frequent_cores(smiles_list, top_k=5, min_atoms=5):
    counter = Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumAtoms() < min_atoms:
                continue
            scaf_smi = Chem.MolToSmiles(scaffold)
            if scaf_smi:
                counter[scaf_smi] += 1
        except Exception:
            continue
    return [core for core, _ in counter.most_common(top_k)]


def sample_dataset_substituents(smiles_list, max_count=300):
    out = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            for frag in frags:
                f_smi = Chem.MolToSmiles(frag)
                if f_smi and 1 <= frag.GetNumAtoms() <= 12:
                    out.add(f_smi)
                if len(out) >= max_count:
                    return list(out)
        except Exception:
            continue
    return list(out)


def compute_properties(mol):
    props = {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "QED": float(QED.qed(mol)),
        "NumRings": int(mol.GetRingInfo().NumRings()),
    }
    try:
        props["SA_Score"] = float(rdMolDescriptors.CalcSAScore(mol))
    except Exception:
        props["SA_Score"] = float(max(1.0, min(10.0, props["MW"] / 100.0)))
    return props


class GenericCorePreserver:
    def __init__(self, core_smiles):
        self.core_smiles = core_smiles
        self.core_mol = Chem.MolFromSmiles(core_smiles)
        if self.core_mol is None:
            raise ValueError(f"Invalid core SMILES: {core_smiles}")

    def validate(self, mol):
        if mol is None:
            return False
        return mol.HasSubstructMatch(self.core_mol)


class MultiCoreMolecularGenerator:
    def __init__(self, core_smiles_list, dataset_substituents=None):
        self.core_smiles_list = core_smiles_list
        self.dataset_substituents = dataset_substituents or []

    def _pick_substituent(self, complexity):
        pool = list(SUBSTITUENT_BANK.get(complexity, SUBSTITUENT_BANK["medium"]))
        pool.extend(self.dataset_substituents[:200])
        if not pool:
            return "C"
        return random.choice(pool)

    def _attach_substituent(self, core_mol, substituent_smi):
        sub_mol = Chem.MolFromSmiles(substituent_smi)
        if sub_mol is None:
            return None

        combo = Chem.CombineMols(core_mol, sub_mol)
        rw = Chem.RWMol(combo)
        core_n = core_mol.GetNumAtoms()
        sub_offset = core_n

        core_attach = [a.GetIdx() for a in core_mol.GetAtoms() if a.GetAtomicNum() > 1 and a.GetDegree() < 4]
        sub_attach = [sub_offset + a.GetIdx() for a in sub_mol.GetAtoms() if a.GetAtomicNum() > 1 and a.GetDegree() < 4]
        if not core_attach or not sub_attach:
            return None

        rw.AddBond(random.choice(core_attach), random.choice(sub_attach), Chem.BondType.SINGLE)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def generate_for_core(self, core_smiles, n=50, complexity="medium", max_attempt_factor=20):
        preserver = GenericCorePreserver(core_smiles)
        core_mol = Chem.MolFromSmiles(core_smiles)
        if core_mol is None:
            return []

        results = []
        attempts = 0
        max_attempts = max(50, n * max_attempt_factor)
        while len(results) < n and attempts < max_attempts:
            attempts += 1
            sub = self._pick_substituent(complexity)
            cand = self._attach_substituent(core_mol, sub)
            if cand is None:
                continue
            if not preserver.validate(cand):
                continue
            smi = Chem.MolToSmiles(cand)
            props = compute_properties(cand)
            results.append({"core_smiles": core_smiles, "smiles": smi, "substituent": sub, **props})
        return results

    def generate_all(self, per_core=50, complexity="medium"):
        all_rows = []
        for core in self.core_smiles_list:
            print(f"[RUN] Core: {core}")
            rows = self.generate_for_core(core, n=per_core, complexity=complexity)
            print(f"      generated={len(rows)}")
            all_rows.extend(rows)
        return all_rows


def save_csv(rows, output_csv):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=None, help="Optional training/reference CSV for auto core detection.")
    parser.add_argument("--core_smiles", action="append", default=None, help="Manual core SMILES; repeatable.")
    parser.add_argument("--top_k_cores", type=int, default=5, help="Number of auto-detected frequent cores.")
    parser.add_argument("--per_core", type=int, default=30, help="Target generated molecules per core.")
    parser.add_argument("--complexity", type=str, default="medium", choices=["simple", "medium", "complex"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non_deterministic_torch", action="store_true",
                        help="Disable torch deterministic mode (not recommended for reproducibility).")
    parser.add_argument("--output_prefix", type=str, default="multi_core_generated")
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic_torch=(not args.non_deterministic_torch))
    print(f"[INFO] Global seed set to {args.seed} (numpy/random/torch).")
    if args.non_deterministic_torch:
        print("[WARN] Torch deterministic mode is disabled by user option.")

    source_smiles = []
    if args.input_csv and os.path.isfile(args.input_csv):
        source_smiles = load_smiles_from_csv(args.input_csv)

    if args.core_smiles:
        cores = list(dict.fromkeys(args.core_smiles))
    elif source_smiles:
        cores = detect_frequent_cores(source_smiles, top_k=args.top_k_cores)
    else:
        cores = DEFAULT_CORES[:args.top_k_cores]

    if not cores:
        raise RuntimeError("No core scaffolds available. Provide --core_smiles or a valid --input_csv.")

    # Keep only RDKit-parseable cores.
    cores = [c for c in cores if Chem.MolFromSmiles(c) is not None]
    if not cores:
        raise RuntimeError("All detected/provided cores are invalid SMILES.")

    dataset_subs = sample_dataset_substituents(source_smiles) if source_smiles else []
    generator = MultiCoreMolecularGenerator(cores, dataset_substituents=dataset_subs)
    rows = generator.generate_all(per_core=args.per_core, complexity=args.complexity)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"{args.output_prefix}_{ts}.csv"
    save_csv(rows, out_csv)

    print(f"[DONE] cores={len(cores)}, total_generated={len(rows)}")
    print(f"[DONE] output_csv={out_csv}")


if __name__ == "__main__":
    main()
