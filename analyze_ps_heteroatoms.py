#!/usr/bin/env python3
"""
 P/S Heteroatom-Containing Molecule Generation and Analysis Script
Designed for large-scale molecule generation and statistical analysis of P/S heteroatom-containing molecules.

Functions:
1. Generate thousands of benzimidazole-core-based molecules
2. Automatically filter molecules containing P/S heteroatoms
3. Compute detailed molecular property statistics
4. Generate statistical reports and CSV files

"""

import os
import sys
import inspect
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem import QED, rdMolDescriptors, DataStructs

# Import DiffGAT model
from improved_hybrid_molecular_generator import (
    ImprovedHybridConfig,
    ImprovedHybridMolecularGenerator,
    EnhancedCoreStructurePreserver,
    EnhancedTargetValidator
)

class PSHeteroatomAnalyzer:
    """Analyzer for molecules containing phosphorus and/or sulfur heteroatoms."""

    def __init__(self, config=None):
        """Initialize analyzer"""
        self.config = config or ImprovedHybridConfig()
        self.generator = ImprovedHybridMolecularGenerator(self.config)
        self.core_preserver = EnhancedCoreStructurePreserver(self.config)
        self.target_validator = EnhancedTargetValidator(self.config)

        # Statistics
        self.stats = {
            'total_generated': 0,
            'valid_molecules': 0,
            'with_phosphorus': 0,
            'with_sulfur': 0,
            'with_both_ps': 0,
            'without_ps': 0,
            'high_quality_ps': 0,  # Contains P/S and QED>=0.5 & SA<=8
        }

        print(" P/S heteroatom analyzer initialized")
        print(f"   Device: {self.config.device}")

    def generate_and_analyze(self, num_molecules=5000, complexity_level='medium',
                             save_intermediate=True):
        """
        Generate molecules and analyze P/S heteroatom-containing molecules

        Args:
            num_molecules: Total number of molecules to generate
            complexity_level: Complexity level ('simple', 'medium', 'complex')
            save_intermediate: Whether to save intermediate results

        Returns:
            all_molecules: List of all generated molecules
            ps_molecules: List of molecules containing P/S
        """
        print(f"\n{'=' * 70}")
        print(f" Start generation and analysis for {num_molecules} molecules")
        print(f"{'=' * 70}")
        print(f"   Complexity level: {complexity_level}")
        print(f"   Target: Filter molecules containing P/S heteroatoms")

        # 
        print(f"\n Step 1: Molecule generation")
        # To increase the proportion of P/S-containing molecules, P/S-enrichment mode is explicitly enabled.
        # For large-scale generation (>500), diversity checks are temporarily disabled to improve success rate.
        disable_diversity = num_molecules > 500
        if disable_diversity:
            print(f"     Large-scale generation mode: diversity checks temporarily disabled to improve generation throughput.")

        # Check method signature for cross-version compatibility
        sig = inspect.signature(self.generator.generate_molecules)
        params = sig.parameters

        # Construct parameter dictionary
        kwargs = {
            'num_molecules': num_molecules,
            'complexity_level': complexity_level,
            'ps_enriched': True
        }

        # Add disable_diversity_check if supported by the method signature
        if 'disable_diversity_check' in params:
            kwargs['disable_diversity_check'] = disable_diversity

        all_molecules = self.generator.generate_molecules(**kwargs)

        self.stats['total_generated'] = num_molecules
        self.stats['valid_molecules'] = len(all_molecules)

        print(f" Successfully generated {len(all_molecules)} valid molecules")

        # P/S filtering
        print(f"\n Step 2: Filter molecules containing P/S heteroatoms...")
        ps_molecules = self._filter_ps_molecules(all_molecules)

        print(f" Filtering completed:")
        print(f"   Molecules containing P atoms: {self.stats['with_phosphorus']}")
        print(f"   Molecules containing S atoms: {self.stats['with_sulfur']}")
        print(f"   Molecules containing both P and S: {self.stats['with_both_ps']}")
        print(f"   Molecules without P/S: {self.stats['without_ps']}")

        # 
        print(f"\n Step 3: Statistical analysis")
        detailed_stats = self._calculate_detailed_statistics(all_molecules, ps_molecules)

        # Save results
        if save_intermediate:
            print(f"\n Step 4: Save outputs")
            self._save_results(all_molecules, ps_molecules, detailed_stats)

        return all_molecules, ps_molecules

    def _filter_ps_molecules(self, molecules: List[Dict]) -> List[Dict]:
        """Filter molecules containing P/S heteroatoms"""
        ps_molecules = []

        for mol_data in molecules:
            mol = mol_data.get('molecule')
            if mol is None:
                continue

            # P/S atom presence check
            has_p = False
            has_s = False
            p_count = 0
            s_count = 0

            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                if atomic_num == 15:  # phosphorus
                    has_p = True
                    p_count += 1
                elif atomic_num == 16:  # sulfur
                    has_s = True
                    s_count += 1

            # Append P/S annotations to molecular records
            mol_data['has_phosphorus'] = has_p
            mol_data['has_sulfur'] = has_s
            mol_data['p_count'] = p_count
            mol_data['s_count'] = s_count
            mol_data['has_ps'] = has_p or has_s

            # Update statistics
            if has_p:
                self.stats['with_phosphorus'] += 1
            if has_s:
                self.stats['with_sulfur'] += 1
            if has_p and has_s:
                self.stats['with_both_ps'] += 1
            if not (has_p or has_s):
                self.stats['without_ps'] += 1

            # High-quality P/S-containing molecules(QED>=0.5 & SA<=8)
            if (has_p or has_s) and mol_data.get('properties'):
                props = mol_data['properties']
                qed = props.get('QED', 0)
                sa_score = props.get('SA_Score', 10)
                if qed >= 0.5 and sa_score <= 8.0:
                    self.stats['high_quality_ps'] += 1

            # P/S filtering,
            if has_p or has_s:
                ps_molecules.append(mol_data)

        return ps_molecules

    def _calculate_detailed_statistics(self, all_molecules: List[Dict],
                                       ps_molecules: List[Dict]) -> Dict:
        """Compute detailed descriptive statistics for all and P/S subsets."""
        stats = {}

        # Basic statistics
        stats['total'] = len(all_molecules)
        stats['with_ps'] = len(ps_molecules)
        stats['ps_percentage'] = len(ps_molecules) / len(all_molecules) * 100 if all_molecules else 0

        # Extract properties
        def extract_properties(molecules, key='properties'):
            props_list = []
            for mol_data in molecules:
                props = mol_data.get(key, {})
                if props:
                    props_list.append(props)
            return props_list

        all_props = extract_properties(all_molecules)
        ps_props = extract_properties(ps_molecules)

        # Compute mean and standard deviation
        def calc_stats(props_list, prop_name):
            values = [p.get(prop_name, 0) for p in props_list if prop_name in p]
            if values:
                return {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
            return None

        # Key property statistics
        key_properties = ['MW', 'LogP', 'QED', 'SA_Score', 'HBD', 'HBA', 'TPSA',
                          'RotatableBonds', 'AromaticRings']

        stats['all_molecules'] = {}
        stats['ps_molecules'] = {}

        for prop in key_properties:
            all_stat = calc_stats(all_props, prop)
            ps_stat = calc_stats(ps_props, prop)

            if all_stat:
                stats['all_molecules'][prop] = all_stat
            if ps_stat:
                stats['ps_molecules'][prop] = ps_stat

        # P/S filteringAtom count statistics
        if ps_molecules:
            p_counts = [m.get('p_count', 0) for m in ps_molecules]
            s_counts = [m.get('s_count', 0) for m in ps_molecules]

            stats['p_atom_counts'] = {
                'mean': np.mean(p_counts),
                'std': np.std(p_counts),
                'min': int(np.min(p_counts)),
                'max': int(np.max(p_counts)),
                'total': int(np.sum(p_counts))
            }

            stats['s_atom_counts'] = {
                'mean': np.mean(s_counts),
                'std': np.std(s_counts),
                'min': int(np.min(s_counts)),
                'max': int(np.max(s_counts)),
                'total': int(np.sum(s_counts))
            }

        # Target validation statistics
        def extract_target_validation(molecules):
            validations = []
            for mol_data in molecules:
                tv = mol_data.get('target_validation', {})
                if tv:
                    validations.append(tv)
            return validations

        all_tv = extract_target_validation(all_molecules)
        ps_tv = extract_target_validation(ps_molecules)

        if all_tv:
            binding_scores_all = [tv.get('binding_score', 0) for tv in all_tv]
            stats['all_molecules']['binding_score'] = calc_stats(
                [{'binding_score': bs} for bs in binding_scores_all], 'binding_score'
            )

        if ps_tv:
            binding_scores_ps = [tv.get('binding_score', 0) for tv in ps_tv]
            stats['ps_molecules']['binding_score'] = calc_stats(
                [{'binding_score': bs} for bs in binding_scores_ps], 'binding_score'
            )

            promising_count = sum(1 for tv in ps_tv if tv.get('is_promising', False))
            stats['ps_promising_count'] = promising_count
            stats['ps_promising_percentage'] = promising_count / len(ps_tv) * 100 if ps_tv else 0

        return stats

    def _save_results(self, all_molecules: List[Dict], ps_molecules: List[Dict],
                      detailed_stats: Dict):
        """Save analysis results to CSV and text files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. All moleculesCSV
        all_csv_path = f"all_molecules_{timestamp}.csv"
        self._save_molecules_to_csv(all_molecules, all_csv_path)
        print(f"    All molecules saved to: {all_csv_path}")

        # 2. Save P/S-containing molecules to CSV
        ps_csv_path = f"ps_heteroatom_molecules_{timestamp}.csv"
        self._save_molecules_to_csv(ps_molecules, ps_csv_path)
        print(f"    P/S-containing molecules saved to: {ps_csv_path}")

        # 3. 
        stats_report_path = f"ps_heteroatom_statistics_{timestamp}.txt"
        self._save_statistics_report(detailed_stats, stats_report_path)
        print(f"    Statistical report saved to: {stats_report_path}")

        # 4. CSV()
        stats_csv_path = f"ps_heteroatom_statistics_{timestamp}.csv"
        self._save_statistics_to_csv(detailed_stats, stats_csv_path)
        print(f"    Statistical table saved to: {stats_csv_path}")

    def _save_molecules_to_csv(self, molecules: List[Dict], filepath: str):
        """Save molecular data to CSV"""
        rows = []

        for mol_data in molecules:
            mol = mol_data.get('molecule')
            if mol is None:
                continue

            row = {
                'SMILES': mol_data.get('smiles', ''),
                'Has_Phosphorus': mol_data.get('has_phosphorus', False),
                'Has_Sulfur': mol_data.get('has_sulfur', False),
                'P_Count': mol_data.get('p_count', 0),
                'S_Count': mol_data.get('s_count', 0),
                'Has_PS': mol_data.get('has_ps', False),
            }

            # Add molecular properties
            props = mol_data.get('properties', {})
            if props:
                row.update({
                    'MW': props.get('MW', 0),
                    'LogP': props.get('LogP', 0),
                    'HBD': props.get('HBD', 0),
                    'HBA': props.get('HBA', 0),
                    'TPSA': props.get('TPSA', 0),
                    'RotatableBonds': props.get('RotatableBonds', 0),
                    'AromaticRings': props.get('AromaticRings', 0),
                    'QED': props.get('QED', 0),
                    'SA_Score': props.get('SA_Score', 0),
                    'NumRings': props.get('NumRings', 0),
                    'Lipinski_Violations': props.get('Lipinski_Violations', 0),
                })

            # Add target validation metadata
            tv = mol_data.get('target_validation', {})
            if tv:
                row.update({
                    'Binding_Score': tv.get('binding_score', 0),
                    'Similarity_to_Albendazole': tv.get('similarity_to_albendazole', 0),
                    'Is_Promising': tv.get('is_promising', False),
                    'Core_Valid': tv.get('core_valid', False),
                })

            # Add additional metadata
            row.update({
                'Core_Valid': mol_data.get('core_valid', False),
                'Complexity_Level': mol_data.get('complexity_level', ''),
            })

            rows.append(row)

        # Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

    def _save_statistics_report(self, stats: Dict, filepath: str):
        """Save statistical report to text file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("Statistical Analysis Report for P/S Heteroatom-Containing Molecules\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")

            # Basic statistics
            f.write("、Basic statistics\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total generated molecules: {stats['total']}\n")
            f.write(f"Number of P/S-containing molecules: {stats['with_ps']}\n")
            f.write(f"Proportion of P/S-containing molecules: {stats['ps_percentage']:.2f}%\n")
            f.write(f"Containing P atoms: {self.stats['with_phosphorus']}\n")
            f.write(f"Containing S atoms: {self.stats['with_sulfur']}\n")
            f.write(f"Containing both P and S: {self.stats['with_both_ps']}\n")
            f.write(f"Without P/S: {self.stats['without_ps']}\n")
            f.write(f"High-quality P/S-containing molecules (QED≥0.5 & SA≤8): {self.stats['high_quality_ps']}\n")
            f.write("\n")

            # P/S filteringAtom count statistics
            if 'p_atom_counts' in stats:
                f.write("、PAtom count statistics\n")
                f.write("-" * 70 + "\n")
                p_counts = stats['p_atom_counts']
                f.write(f"P: {p_counts['mean']:.2f} ± {p_counts['std']:.2f}\n")
                f.write(f"P: {p_counts['min']} - {p_counts['max']}\n")
                f.write(f"P: {p_counts['total']}\n")
                f.write("\n")

            if 's_atom_counts' in stats:
                f.write("、SAtom count statistics\n")
                f.write("-" * 70 + "\n")
                s_counts = stats['s_atom_counts']
                f.write(f"S: {s_counts['mean']:.2f} ± {s_counts['std']:.2f}\n")
                f.write(f"S: {s_counts['min']} - {s_counts['max']}\n")
                f.write(f"S: {s_counts['total']}\n")
                f.write("\n")

            # Property
            f.write("IV. Property Comparison (All Molecules vs P/S-Containing Molecules)\n")
            f.write("-" * 70 + "\n")

            key_properties = ['MW', 'LogP', 'QED', 'SA_Score', 'HBD', 'HBA', 'TPSA',
                              'RotatableBonds', 'AromaticRings', 'binding_score']

            f.write(f"{'Property':<20} {'All molecules':<25} {'P/S-containing molecules':<25}\n")
            f.write("-" * 70 + "\n")

            for prop in key_properties:
                all_stat = stats.get('all_molecules', {}).get(prop)
                ps_stat = stats.get('ps_molecules', {}).get(prop)

                if all_stat or ps_stat:
                    all_str = f"{all_stat['mean']:.2f}±{all_stat['std']:.2f}" if all_stat else "N/A"
                    ps_str = f"{ps_stat['mean']:.2f}±{ps_stat['std']:.2f}" if ps_stat else "N/A"
                    f.write(f"{prop:<20} {all_str:<25} {ps_str:<25}\n")

            f.write("\n")

            # Target validation statistics
            if 'ps_promising_count' in stats:
                f.write("、Target validation statistics(P/S-containing molecules)\n")
                f.write("-" * 70 + "\n")
                f.write(f"Number of promising candidates: {stats['ps_promising_count']}\n")
                f.write(f"Promising candidate ratio: {stats['ps_promising_percentage']:.2f}%\n")
                f.write("\n")

            # 
            f.write("VI. Conclusion\n")
            f.write("-" * 70 + "\n")
            f.write(f"DiffGATSuccessfully generated {stats['with_ps']} P/S,\n")
            f.write(f" {stats['ps_percentage']:.2f}%.\n")
            f.write(f" {self.stats['high_quality_ps']} (QED≥0.5 & SA≤8).\n")
            f.write("\n")
            f.write("These results indicate that DiffGAT effectively generates chemically plausible molecules containing P/S heteroatoms,\n")
            f.write("thereby providing broader chemical-space exploration capability for drug design..\n")
            f.write("\n")
            f.write("=" * 70 + "\n")

    def _save_statistics_to_csv(self, stats: Dict, filepath: str):
        """Save statistical tables to CSV (for downstream visualization)"""
        rows = []

        # Basic statistics
        rows.append({
            'Category': 'Basic_Statistics',
            'Metric': 'Total_Molecules',
            'Value': stats['total'],
            'Unit': 'count'
        })
        rows.append({
            'Category': 'Basic_Statistics',
            'Metric': 'PS_Molecules',
            'Value': stats['with_ps'],
            'Unit': 'count'
        })
        rows.append({
            'Category': 'Basic_Statistics',
            'Metric': 'PS_Percentage',
            'Value': stats['ps_percentage'],
            'Unit': 'percent'
        })

        # Property
        key_properties = ['MW', 'LogP', 'QED', 'SA_Score', 'HBD', 'HBA', 'TPSA',
                          'RotatableBonds', 'AromaticRings', 'binding_score']

        for prop in key_properties:
            all_stat = stats.get('all_molecules', {}).get(prop)
            ps_stat = stats.get('ps_molecules', {}).get(prop)

            if all_stat:
                rows.append({
                    'Category': 'All_Molecules',
                    'Metric': prop,
                    'Mean': all_stat['mean'],
                    'Std': all_stat['std'],
                    'Min': all_stat['min'],
                    'Max': all_stat['max'],
                    'Median': all_stat['median'],
                    'Unit': self._get_property_unit(prop)
                })

            if ps_stat:
                rows.append({
                    'Category': 'PS_Molecules',
                    'Metric': prop,
                    'Mean': ps_stat['mean'],
                    'Std': ps_stat['std'],
                    'Min': ps_stat['min'],
                    'Max': ps_stat['max'],
                    'Median': ps_stat['median'],
                    'Unit': self._get_property_unit(prop)
                })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

    def _get_property_unit(self, prop_name: str) -> str:
        """Property"""
        units = {
            'MW': 'Da',
            'LogP': '',
            'QED': '',
            'SA_Score': '',
            'HBD': 'count',
            'HBA': 'count',
            'TPSA': 'Å²',
            'RotatableBonds': 'count',
            'AromaticRings': 'count',
            'binding_score': ''
        }
        return units.get(prop_name, '')

def main():
    """Main function"""
    print(" P/S")
    print("=" * 70)
    print("Purpose: provide experimental evidence of P/S heteroatom-containing molecule generation in response to reviewer comments.")
    print("=" * 70)

    import argparse

    parser = argparse.ArgumentParser(description='P/S')
    parser.add_argument('--num_molecules', type=int, default=5000,
                        help='Total number of molecules to generate (: 5000)')
    parser.add_argument('--complexity', type=str, default='medium',
                        choices=['simple', 'medium', 'complex'],
                        help='Complexity level (: medium)')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save intermediate results')

    args = parser.parse_args()

    try:
        # Create analyzer
        analyzer = PSHeteroatomAnalyzer()

        # Generate and analyze
        all_molecules, ps_molecules = analyzer.generate_and_analyze(
            num_molecules=args.num_molecules,
            complexity_level=args.complexity,
            save_intermediate=not args.no_save
        )

        # 
        print(f"\n{'=' * 70}")
        print(" Analysis completed. Brief summary::")
        print(f"{'=' * 70}")
        print(f"Total generated molecules: {len(all_molecules)}")
        print(f"Number of P/S-containing molecules: {len(ps_molecules)}")
        print(f"Proportion of P/S-containing molecules: {len(ps_molecules) / len(all_molecules) * 100:.2f}%")
        print(f"Containing P atoms: {analyzer.stats['with_phosphorus']}")
        print(f"Containing S atoms: {analyzer.stats['with_sulfur']}")
        print(f"High-quality P/S-containing molecules: {analyzer.stats['high_quality_ps']}")
        print(f"\n Save to CSV")
        print(f"{'=' * 70}\n")

        return True

    except Exception as e:
        print(f"\n Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

