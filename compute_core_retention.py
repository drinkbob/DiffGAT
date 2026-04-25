"""
compute_core_retention.py

:
1.  CSV( SMILES)；
2.  EnhancedCoreStructurePreserver  SMARTS ；
3.  scaffold retention rate, CSV .

( DiffGAT):
    python compute_core_retention.py --csv diffgat_generated_benzimidazole.csv --model DiffGAT

():
    python compute_core_retention.py \
        --csv diffgat_generated_benzimidazole.csv --model DiffGAT \
        --csv_baseline mds_results.csv MDM \
        --csv_baseline moldiff_results.csv MolDiff
"""

import argparse
import pandas as pd
from rdkit import Chem

# ,
from improved_hybrid_molecular_generator import (
    EnhancedCoreStructurePreserver,
    ImprovedHybridConfig,
)

BENZIMIDAZOLE_SMARTS = "c1ccc2[nH]cnc2c1"

def build_core_preserver(core_smiles: str = BENZIMIDAZOLE_SMARTS):
    """ EnhancedCoreStructurePreserver ."""
    config = ImprovedHybridConfig()
    config.core_smiles = core_smiles
    return EnhancedCoreStructurePreserver(config)

def validate_core(mol, preserver) -> bool:
    """,validation."""
    if mol is None:
        return False
    try:
        core_valid, msg = preserver.validate_core_integrity(mol)
        return bool(core_valid)
    except Exception:
        return False

def compute_retention_for_csv(csv_path: str,
                              model_name: str,
                              smiles_column: str = "SMILES",
                              use_preserver: bool = True,
                              core_smarts: str = BENZIMIDAZOLE_SMARTS):
    """ CSV  scaffold retention rate, DataFrame ."""
    df = pd.read_csv(csv_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in {csv_path}. "
                         f"Available columns: {list(df.columns)}")

    smiles_list = df[smiles_column].astype(str).tolist()
    n_total = 0
    n_core_valid = 0

    if use_preserver:
        preserver = build_core_preserver(core_smarts)
        use_smarts = False
    else:
        preserver = None
        use_smarts = True
        core_mol = Chem.MolFromSmarts(core_smarts)

    core_flags = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            core_flags.append(False)
            continue

        n_total += 1

        if use_smarts:
            has_core = mol.HasSubstructMatch(core_mol)
            core_flags.append(has_core)
            if has_core:
                n_core_valid += 1
        else:
            has_core = validate_core(mol, preserver)
            core_flags.append(has_core)
            if has_core:
                n_core_valid += 1

    retention = n_core_valid / max(1, n_total)
    df[f"{model_name}_core_valid"] = core_flags

    print(f"[{model_name}] core retention: {n_core_valid}/{n_total} = {retention:.4f}")
    return df, retention

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="DiffGAT  CSV ( SMILES )")
    parser.add_argument("--model", default="DiffGAT",
                        help=" CSV ,")
    parser.add_argument("--smiles_col", default="SMILES",
                        help="SMILES ( 'SMILES')")
    parser.add_argument("--no_preserver", action="store_true",
                        help=" EnhancedCoreStructurePreserver, SMARTS ")
    parser.add_argument("--csv_baseline", nargs=2, action="append",
                        metavar=("CSV_PATH", "MODEL_NAME"),
                        help=" CSV ,")
    parser.add_argument("--out_prefix", default="core_retention",
                        help="")
    args = parser.parse_args()

    use_preserver = not args.no_preserver

    # DiffGAT
    df_diffgat, r_diffgat = compute_retention_for_csv(
        csv_path=args.csv,
        model_name=args.model,
        smiles_column=args.smiles_col,
        use_preserver=use_preserver,
    )
    out_path = f"{args.out_prefix}_{args.model}.csv"
    df_diffgat.to_csv(out_path, index=False)
    print(f"Saved DiffGAT annotated CSV to: {out_path}")

    retention_dict = {args.model: r_diffgat}

    # ()
    if args.csv_baseline:
        for csv_path, model_name in args.csv_baseline:
            df_base, r_base = compute_retention_for_csv(
                csv_path=csv_path,
                model_name=model_name,
                smiles_column=args.smiles_col,
                use_preserver=use_preserver,
            )
            out_path = f"{args.out_prefix}_{model_name}.csv"
            df_base.to_csv(out_path, index=False)
            print(f"Saved {model_name} annotated CSV to: {out_path}")
            retention_dict[model_name] = r_base

    # 
    stat_df = pd.DataFrame(
        {"Model": list(retention_dict.keys()),
         "CoreRetention": [retention_dict[m] for m in retention_dict]}
    )
    stat_df.to_csv(f"{args.out_prefix}_summary.csv", index=False)
    print(stat_df)

if __name__ == "__main__":
    main()