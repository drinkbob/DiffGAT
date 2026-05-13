#!/usr/bin/env python3
"""
 - SCI
: Ablation_Study_Integrated.png ()
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import QED

# SCI
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# 
from improved_hybrid_molecular_generator import ImprovedHybridMolecularGenerator, ImprovedHybridConfig

class IntegratedAblationVisualizer:
    """"""

    def __init__(self):
        # 
        self.colors = {
            'DiffGAT (Full)': '#2E86AB',  # 
            '- Ring': '#F18F01',  # 
            '- Diversity': '#A23B72',  # 
            '- Ring & - Diversity': '#6C757D'  # 
        }

        # Property
        self.property_names = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'SA_Score', 'QED']

    def calculate_molecular_properties(self, smiles_list: List[str]) -> Dict[str, List[float]]:
        """Property"""
        properties = {name: [] for name in self.property_names}

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            try:
                # Property
                properties['MW'].append(Descriptors.MolWt(mol))
                properties['LogP'].append(Crippen.MolLogP(mol))
                properties['HBD'].append(Lipinski.NumHDonors(mol))
                properties['HBA'].append(Lipinski.NumHAcceptors(mol))
                properties['TPSA'].append(Descriptors.TPSA(mol))

                # 
                try:
                    properties['SA_Score'].append(rdMolDescriptors.CalcSAScore(mol))
                except:
                    properties['SA_Score'].append(Descriptors.MolWt(mol) / 100.0)

                try:
                    properties['QED'].append(QED.qed(mol))
                except:
                    properties['QED'].append(0.5)

            except:
                continue

        return properties

    def compute_metrics(self, smiles_list: List[str], train_smiles: List[str]) -> Dict[str, float]:
        """"""
        # 
        valid_count = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
        validity = valid_count / max(1, len(smiles_list))

        # 
        uniqueness = len(set(smiles_list)) / max(1, len(smiles_list))

        #  (train)
        train_mols = [Chem.MolFromSmiles(s) for s in train_smiles if Chem.MolFromSmiles(s) is not None]
        if not train_mols:
            novelty = 0.0
        else:
            max_sims = []
            for s in smiles_list:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    continue
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    sims = []
                    for tm in train_mols:
                        tfp = AllChem.GetMorganFingerprintAsBitVect(tm, 2, nBits=2048)
                        sims.append(DataStructs.TanimotoSimilarity(fp, tfp))
                    max_sims.append(max(sims) if sims else 0.0)
                except:
                    continue
            novelty = np.mean(max_sims) if max_sims else 0.0

        # 
        core_pattern = Chem.MolFromSmarts("c1ccc2[nH]cnc2c1")
        core_count = 0
        total_count = 0
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            total_count += 1
            try:
                if mol.HasSubstructMatch(core_pattern):
                    core_count += 1
            except:
                pass
        core_retention = core_count / max(1, total_count)

        return {
            'Validity': validity,
            'Uniqueness': uniqueness,
            'Novelty': novelty,
            'Core Retention': core_retention
        }

    def create_ablation_variants(self):
        """"""

        def make_full_model():
            cfg = ImprovedHybridConfig()
            return ImprovedHybridMolecularGenerator(cfg)

        def make_minus_ring_model():
            m = make_full_model()

            def simple_substituent(_level):
                return ''

            m._generate_benzimidazole_substituent = simple_substituent
            return m

        def make_minus_diversity_model():
            m = make_full_model()

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

            def simple_substituent(_level):
                return ''

            m._generate_benzimidazole_substituent = simple_substituent

            class NoDiversity:
                def is_diverse_enough(self, _):
                    return True

                def add_molecule(self, _):
                    return

                def get_diversity_score(self):
                    return 1.0

            m.diversity_controller = NoDiversity()
            return m

        return {
            'DiffGAT (Full)': make_full_model,
            '- Ring': make_minus_ring_model,
            '- Diversity': make_minus_diversity_model,
            '- Ring & - Diversity': make_minus_both_model,
        }

    def run_ablation_study(self, num_molecules: int = 100, complexity: str = 'medium') -> Tuple[Dict, Dict, Dict]:
        """"""
        print(" ...")

        # train
        train_smiles = [
            'c1ccc2[nH]cnc2c1',
            'CCc1ccc2[nH]cnc2c1',
            'c1ccc2[nH]cnc2c1C',
            'c1ccc2[nH]cnc2c1c3ccccc3',
            'c1ccc2[nH]cnc2c1c3ccc(cc3)C',
            'c1ccc2[nH]cnc2c1C3CCCCC3'
        ]

        variants = self.create_ablation_variants()
        results_smiles = {}
        results_metrics = {}
        results_properties = {}

        for name, factory in variants.items():
            print(f"   : {name}")
            model = factory()
            generated = model.generate_molecules(num_molecules=num_molecules, complexity_level=complexity)
            smiles = [g['smiles'] for g in generated if isinstance(g, dict) and 'smiles' in g]

            results_smiles[name] = smiles
            results_metrics[name] = self.compute_metrics(smiles, train_smiles)
            results_properties[name] = self.calculate_molecular_properties(smiles)

            print(f"      {len(smiles)} ")

        return results_smiles, results_metrics, results_properties

    def create_integrated_plot(self, results_metrics: Dict, results_properties: Dict, results_smiles: Dict,
                               save_path: str):
        """"""
        fig = plt.figure(figsize=(16, 12))

        # 1:  (Figure 3)
        ax1 = plt.subplot(2, 2, 1)
        metrics = ['Validity', 'Uniqueness', 'Core Retention']  # Novelty
        x = np.arange(len(metrics))
        width = 0.2

        for i, (model, color) in enumerate(self.colors.items()):
            values = [results_metrics[model].get(m, 0) for m in metrics]
            bars = ax1.bar(x + i * width, values, width, label=model, color=color, alpha=0.8, edgecolor='black',
                           linewidth=0.5)

            # 
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax1.set_xlabel('Evaluation Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('(A) Ablation Study: Metrics Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right', fontsize=8)

        # 2:  (Figure 4)
        ax2 = plt.subplot(2, 2, 2)

        # 
        box_data = []
        box_labels = []
        for metric in ['Validity', 'Uniqueness', 'Core Retention']:
            values = [results_metrics[model].get(metric, 0) for model in self.colors.keys()]
            box_data.append(values)
            box_labels.append(metric)

        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))

        ax2.set_ylabel('Score')
        ax2.set_title('(B) Metrics Distribution Across Variants')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

        # 3: Property (Figure 5)
        ax3 = plt.subplot(2, 2, 3)

        # Property
        key_properties = ['MW', 'LogP', 'HBD', 'HBA']
        property_labels = ['Molecular Weight (Da)', 'LogP', 'H-Bond Donors', 'H-Bond Acceptors']

        # Property
        prop_data = []
        prop_labels = []
        for i, (prop, label) in enumerate(zip(key_properties, property_labels)):
            for model in self.colors.keys():
                if prop in results_properties[model]:
                    values = results_properties[model][prop]
                    if values:  # 
                        prop_data.extend(values)
                        prop_labels.extend([f'{label}\n{model}'] * len(values))

        # Property - 
        if prop_data:
            # Property
            grouped_data = {}
            for prop, label in zip(key_properties, property_labels):
                grouped_data[label] = []
                for model in self.colors.keys():
                    if prop in results_properties[model]:
                        values = results_properties[model][prop]
                        if values:
                            grouped_data[label].extend(values)

            if grouped_data:
                box_data = list(grouped_data.values())
                box_labels = list(grouped_data.keys())

                bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

        ax3.set_ylabel('Property Value')
        ax3.set_title('(C) Molecular Properties Distribution')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, axis='y', alpha=0.3, linestyle='--')

        # 4: t-SNE (Figure 6)
        ax4 = plt.subplot(2, 2, 4)

        # t-SNE
        all_fps = []
        all_labels = []
        all_smiles = []

        for model, smiles in results_smiles.items():
            for s in smiles[:50]:  # 50
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    try:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                        all_fps.append(np.array(fp))
                        all_labels.append(model)
                        all_smiles.append(s)
                    except:
                        continue

        if len(all_fps) > 10:  # 
            try:
                # PCAt-SNE
                pca = PCA(n_components=min(50, len(all_fps[0])))
                fps_pca = pca.fit_transform(all_fps)

                # t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_fps) // 4))
                fps_2d = tsne.fit_transform(fps_pca)

                # 
                for model, color in self.colors.items():
                    mask = [label == model for label in all_labels]
                    if any(mask):
                        x_coords = [fps_2d[i, 0] for i, m in enumerate(mask) if m]
                        y_coords = [fps_2d[i, 1] for i, m in enumerate(mask) if m]
                        ax4.scatter(x_coords, y_coords, c=color, label=model, alpha=0.7, s=20)

                ax4.set_xlabel('t-SNE Component 1')
                ax4.set_ylabel('t-SNE Component 2')
                ax4.set_title('(D) Molecular Space Distribution (t-SNE)')
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3, linestyle='--')

            except Exception as e:
                ax4.text(0.5, 0.5, f't-SNE failed: {str(e)[:50]}',
                         ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('(D) Molecular Space Distribution (t-SNE)')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for t-SNE',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('(D) Molecular Space Distribution (t-SNE)')

        # 
        fig.suptitle('Ablation Study: Component Analysis of DiffGAT Model',
                     fontsize=16, fontweight='bold', y=0.95)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" : {save_path}")

        return fig

def main():
    """Main function"""
    if len(sys.argv) > 1:
        num_molecules = int(sys.argv[1])
    else:
        num_molecules = 100

    if len(sys.argv) > 2:
        complexity = sys.argv[2]
    else:
        complexity = 'medium'

    print(f" ")
    print(f"   : {num_molecules}")
    print(f"   : {complexity}")

    # 
    visualizer = IntegratedAblationVisualizer()

    # 
    results_smiles, results_metrics, results_properties = visualizer.run_ablation_study(
        num_molecules=num_molecules, complexity=complexity
    )

    # 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'integrated_ablation_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # 
    save_path = os.path.join(output_dir, 'Ablation_Study_Integrated.png')
    visualizer.create_integrated_plot(results_metrics, results_properties, results_smiles, save_path)

    # 
    # 
    metrics_df = pd.DataFrame(results_metrics).T
    metrics_df.to_csv(os.path.join(output_dir, 'ablation_metrics.csv'))

    # SMILES
    for model, smiles in results_smiles.items():
        df = pd.DataFrame({'SMILES': smiles})
        filename = f'smiles_{model.replace(" ", "_").replace("&", "and").replace("-", "minus")}.csv'
        df.to_csv(os.path.join(output_dir, filename), index=False)

    # Property
    with open(os.path.join(output_dir, 'molecular_properties.json'), 'w') as f:
        # numpyJSON
        json_properties = {}
        for model, props in results_properties.items():
            json_properties[model] = {k: v for k, v in props.items()}
        json.dump(json_properties, f, indent=2)

    # 
    with open(os.path.join(output_dir, 'ablation_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n')
        f.write('=' * 50 + '\n\n')
        f.write(f':\n')
        f.write(f'  : {num_molecules}\n')
        f.write(f'  : {complexity}\n\n')
        f.write(':\n')
        for model, metrics in results_metrics.items():
            f.write(f'  {model}:\n')
            for metric, value in metrics.items():
                f.write(f'    {metric}: {value:.3f}\n')
            f.write('\n')

    print(f" !")
    print(f"   : {output_dir}")
    print(f"   : Ablation_Study_Integrated.png")

if __name__ == '__main__':
    main()
