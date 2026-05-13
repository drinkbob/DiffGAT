#!/usr/bin/env python3
"""
 evaluation and visualization scripts
Molecular Generation Evaluation and Visualization Script

Functions:
1. 
2. (Validity、Uniqueness、Novelty、Core Retention)
3. 
4. t-SNE
5. 
6. 

:
python molecular_evaluation_visualization.py [num_molecules] [complexity_level]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import DataStructs, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import os
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

#  - SCI
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# 
try:
    from improved_hybrid_molecular_generator import ImprovedHybridMolecularGenerator, ImprovedHybridConfig

    print(" ")
except ImportError as e:
    print(f" : {e}")
    print(" improved_hybrid_molecular_generator.py ")
    sys.exit(1)

class MolecularEvaluator:
    """ - """

    def __init__(self, core_smarts="c1ccc2[nH]cnc2c1"):
        self.core_smarts = core_smarts
        self.core_pattern = Chem.MolFromSmarts(core_smarts)

    def calculate_validity(self, smiles_list):
        """ - RDKit"""
        if not smiles_list:
            return 0.0

        valid_count = 0
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
            except:
                continue

        return valid_count / len(smiles_list)

    def calculate_uniqueness(self, smiles_list):
        """ - """
        if not smiles_list:
            return 0.0

        # SMILES
        valid_smiles = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(smiles)
            except:
                continue

        if not valid_smiles:
            return 0.0

        unique_smiles = set(valid_smiles)
        return len(unique_smiles) / len(valid_smiles)

    def calculate_novelty(self, generated_smiles, train_smiles):
        """ - trainTanimoto"""
        if not generated_smiles or not train_smiles:
            return 0.0

        # 
        valid_generated = []
        valid_train = []

        for smiles in generated_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_generated.append(mol)
            except:
                continue

        for smiles in train_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_train.append(mol)
            except:
                continue

        if not valid_generated or not valid_train:
            return 0.0

        # train
        similarities = []
        for gen_mol in valid_generated:
            mol_similarities = []
            for train_mol in valid_train:
                try:
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
                    fp2 = AllChem.GetMorganFingerprintAsBitVect(train_mol, 2, nBits=2048)
                    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                    mol_similarities.append(similarity)
                except:
                    continue

            if mol_similarities:
                similarities.append(max(mol_similarities))  # 

        if not similarities:
            return 0.0

        # ()
        return np.mean(similarities)

    def calculate_core_retention(self, smiles_list):
        """ - """
        if not smiles_list:
            return 0.0

        if self.core_pattern is None:
            return 0.0

        retained_count = 0
        total_count = 0

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    total_count += 1
                    if mol.HasSubstructMatch(self.core_pattern):
                        retained_count += 1
            except:
                continue

        return retained_count / total_count if total_count > 0 else 0.0

class MolecularVisualizer:
    """ - t-SNE"""

    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def smiles_to_fingerprint(self, smiles_list, radius=2, nBits=2048):
        """SMILES"""
        fingerprints = []
        valid_smiles = []

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
                    fingerprints.append(np.array(fp))
                    valid_smiles.append(smiles)
            except:
                continue

        return np.array(fingerprints), valid_smiles

    def perform_tsne(self, fingerprints, perplexity=30, n_components=2):
        """t-SNE"""
        if len(fingerprints) < 2:
            print(" ,t-SNE")
            return None

        # perplexity
        perplexity = min(perplexity, len(fingerprints) - 1)
        if perplexity < 5:
            perplexity = 5

        try:
            tsne = TSNE(n_components=n_components, perplexity=perplexity,
                        random_state=42, n_iter=1000, learning_rate='auto')
            embeddings = tsne.fit_transform(fingerprints)
            return embeddings
        except Exception as e:
            print(f"t-SNE,PCA: {e}")
            try:
                pca = PCA(n_components=n_components)
                embeddings = pca.fit_transform(fingerprints)
                return embeddings
            except Exception as e2:
                print(f"PCA: {e2}")
                return None

    def plot_tsne(self, embeddings, labels, title="Molecular Space Distribution", save_path=None):
        """t-SNE"""
        if embeddings is None:
            print(" Cannot plot t-SNE: No valid embeddings")
            return

        plt.figure(figsize=(10, 8))

        # 
        unique_labels = list(set(labels))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

        # 
        label_mapping = {
            'train': 'Training Set',
            '': 'Your Model',
            '': 'Baseline Model'
        }

        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            color = self.colors[i % len(self.colors)]
            marker = markers[i % len(markers)]

            # 
            english_label = label_mapping.get(label, label)

            plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                        c=color, marker=marker, label=english_label, alpha=0.7, s=50)

        plt.title(title, fontsize=14, fontweight='bold', fontfamily='Times New Roman')
        plt.xlabel('t-SNE Component 1', fontsize=12, fontfamily='Times New Roman')
        plt.ylabel('t-SNE Component 2', fontsize=12, fontfamily='Times New Roman')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" t-SNE: {save_path}")

        plt.show()

    def plot_comparison_bars(self, metrics_data, save_path=None):
        """"""
        plt.figure(figsize=(10, 6))

        # 
        models = list(metrics_data.keys())
        metrics = ['Validity', 'Uniqueness', 'Novelty', 'Core Retention']

        # 
        model_mapping = {
            '': 'Your Model',
            '': 'Baseline Model'
        }

        x = np.arange(len(metrics))
        width = 0.35

        # 
        for i, model in enumerate(models):
            values = [metrics_data[model].get(metric, 0) for metric in metrics]
            color = self.colors[i % len(self.colors)]
            english_model = model_mapping.get(model, model)
            plt.bar(x + i * width, values, width, label=english_model, color=color, alpha=0.8)

        plt.xlabel('Evaluation Metrics', fontsize=12, fontfamily='Times New Roman')
        plt.ylabel('Score', fontsize=12, fontfamily='Times New Roman')
        plt.title('Molecular Generation Model Comparison', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
        plt.xticks(x + width / 2, metrics, fontsize=11, fontfamily='Times New Roman')
        plt.legend(fontsize=10, frameon=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(0, 1)

        # 
        for i, model in enumerate(models):
            values = [metrics_data[model].get(metric, 0) for metric in metrics]
            for j, v in enumerate(values):
                plt.text(j + i * width, v + 0.01, f'{v:.3f}',
                         ha='center', va='bottom', fontsize=9, fontfamily='Times New Roman')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" : {save_path}")

        plt.show()

def generate_baseline_molecules(num_molecules=100):
    """()"""
    print(f"  ({num_molecules} )...")

    # ()
    baseline_templates = [
        # 
        "C", "CC", "CCC", "CCCC", "CCCCC",
        "c1ccccc1",  # benzene
        "C1CCCCC1",  # cyclohexane
        "c1ccncc1",  # pyridine

        # ()
        "c1ccc2[nH]cnc2c1",  # 
        "c1ccc2[nH]cc2c1",  # 
        "c1ccc2ccnc2c1",  # 

        # 
        "c1ccsc1",  # thiophene
        "c1ccoc1",  # furan
        "c1c[nH]cc1",  # 

        # 
        "c1ccc2ccccc2c1",  # naphthalene
        "c1ccc2c(c1)cccc2",  # naphthalene
        "c1ccc2[nH]cnc2c1C",  # 
    ]

    generated_smiles = []

    for i in range(num_molecules):
        # 
        template = np.random.choice(baseline_templates)

        # 
        substituents = ['C', 'CC', 'CCC', 'F', 'Cl', 'O', 'N']
        if np.random.random() < 0.3:  # 30%
            substituent = np.random.choice(substituents)
            modified_smiles = f"{template}{substituent}"
        else:
            modified_smiles = template

        # validation
        try:
            mol = Chem.MolFromSmiles(modified_smiles)
            if mol is not None:
                generated_smiles.append(modified_smiles)
        except:
            # ,
            try:
                mol = Chem.MolFromSmiles(template)
                if mol is not None:
                    generated_smiles.append(template)
            except:
                continue

    print(f"  {len(generated_smiles)} ")
    return generated_smiles

def generate_training_set():
    """train()"""
    print(" train...")

    # train
    train_templates = [
        "c1ccc2[nH]cnc2c1",  # 
        "CCc1ccc2[nH]cnc2c1",  # Albendazole
        "c1ccc2[nH]cnc2c1C",  # 
        "c1ccc2[nH]cnc2c1c3ccccc3",  # 
        "c1ccc2[nH]cnc2c1C3CCCCC3",  # 
        "c1ccc2[nH]cnc2c1c3ccc(cc3)C",  # 
        "c1ccc2[nH]cnc2c1c3ccc(cc3)O",  # 
        "c1ccc2[nH]cnc2c1c3ccc(cc3)N",  # 
        "c1ccc2[nH]cnc2c1c3ccc(cc3)F",  # 
        "c1ccc2[nH]cnc2c1c3ccc(cc3)Cl",  # 
        "c1ccc2[nH]cnc2c1C3CCCCCC3",  # 
        "c1ccc2[nH]cnc2c1c3ccc4ccccc4c3",  # naphthalene
        "c1ccc2[nH]cnc2c1c3ccc4c(c3)cccc4",  # 
    ]

    # 
    train_smiles = []
    for template in train_templates:
        train_smiles.append(template)

        # 
        for _ in range(5):
            try:
                mol = Chem.MolFromSmiles(template)
                if mol:
                    # 
                    substituents = ['C', 'CC', 'F', 'Cl', 'O', 'N']
                    if np.random.random() < 0.5:
                        substituent = np.random.choice(substituents)
                        modified_smiles = f"{template}{substituent}"
                        test_mol = Chem.MolFromSmiles(modified_smiles)
                        if test_mol:
                            train_smiles.append(modified_smiles)
            except:
                continue

    print(f" train {len(train_smiles)} ")
    return train_smiles

def main():
    """Main function - """
    print(" ")
    print("=" * 60)

    # 
    num_molecules = 100
    complexity_level = 'medium'

    if len(sys.argv) > 1:
        try:
            num_molecules = int(sys.argv[1])
        except ValueError:
            print(f" ,: {num_molecules}")

    if len(sys.argv) > 2:
        complexity_level = sys.argv[2]
        if complexity_level not in ['simple', 'medium', 'complex']:
            print(f" ,: {complexity_level}")
            complexity_level = 'medium'

    print(f" :")
    print(f"   : {num_molecules}")
    print(f"   Complexity level: {complexity_level}")

    # 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"evaluation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. 
        print("\n ...")
        config = ImprovedHybridConfig()
        your_model = ImprovedHybridMolecularGenerator(config)
        print(" ")

        # 2. 
        print(f"\n ...")
        your_molecules = your_model.generate_molecules(
            num_molecules=num_molecules,
            complexity_level=complexity_level
        )

        if not your_molecules:
            print(" ")
            return

        your_smiles = [mol['smiles'] for mol in your_molecules if mol.get('smiles')]
        print(f"  {len(your_smiles)} ")

        # 3. 
        print(f"\n ...")
        baseline_smiles = generate_baseline_molecules(num_molecules)

        # 4. train
        print(f"\n train...")
        train_smiles = generate_training_set()

        # 5. 
        print(f"\n ...")
        evaluator = MolecularEvaluator()
        visualizer = MolecularVisualizer()

        # 6. 
        print(f"\n ...")

        # 
        your_metrics = {
            'Validity': evaluator.calculate_validity(your_smiles),
            'Uniqueness': evaluator.calculate_uniqueness(your_smiles),
            'Novelty': evaluator.calculate_novelty(your_smiles, train_smiles),
            'Core Retention': evaluator.calculate_core_retention(your_smiles)
        }

        # 
        baseline_metrics = {
            'Validity': evaluator.calculate_validity(baseline_smiles),
            'Uniqueness': evaluator.calculate_uniqueness(baseline_smiles),
            'Novelty': evaluator.calculate_novelty(baseline_smiles, train_smiles),
            'Core Retention': evaluator.calculate_core_retention(baseline_smiles)
        }

        # 7. 
        print(f"\n :")
        print(f"{'':<15} {'':<12} {'':<12} {'':<10}")
        print("-" * 50)

        metrics_data = {
            '': your_metrics,
            '': baseline_metrics
        }

        for metric in ['Validity', 'Uniqueness', 'Novelty', 'Core Retention']:
            your_score = your_metrics[metric]
            baseline_score = baseline_metrics[metric]

            if metric == 'Novelty':
                # 
                advantage = "" if your_score < baseline_score else ""
            else:
                # 
                advantage = "" if your_score > baseline_score else ""

            print(f"{metric:<15} {your_score:<12.3f} {baseline_score:<12.3f} {advantage:<10}")

        # 8. t-SNE
        print(f"\n t-SNE...")

        # 
        all_smiles = train_smiles + your_smiles + baseline_smiles
        all_labels = ['Training Set'] * len(train_smiles) + ['Your Model'] * len(your_smiles) + [
            'Baseline Model'] * len(baseline_smiles)

        # 
        fingerprints, valid_smiles = visualizer.smiles_to_fingerprint(all_smiles)

        if len(fingerprints) > 0:
            # 
            valid_labels = [label for i, label in enumerate(all_labels) if all_smiles[i] in valid_smiles]

            # t-SNE
            embeddings = visualizer.perform_tsne(fingerprints)

            if embeddings is not None:
                # t-SNE
                tsne_path = os.path.join(output_dir, "molecular_space_tsne.png")
                visualizer.plot_tsne(embeddings, valid_labels,
                                     title="Molecular Chemical Space Distribution (t-SNE)",
                                     save_path=tsne_path)

        # 9. 
        print(f"\n ...")
        comparison_path = os.path.join(output_dir, "metrics_comparison.png")
        visualizer.plot_comparison_bars(metrics_data, save_path=comparison_path)

        # 10. 
        print(f"\n ...")

        # SMILES
        results_df = pd.DataFrame({
            'Model': ['Your Model'] * len(your_smiles) + ['Baseline Model'] * len(baseline_smiles),
            'SMILES': your_smiles + baseline_smiles
        })

        results_path = os.path.join(output_dir, "generated_molecules.csv")
        results_df.to_csv(results_path, index=False, encoding='utf-8')

        # 
        metrics_df = pd.DataFrame(metrics_data).T
        metrics_path = os.path.join(output_dir, "evaluation_metrics.csv")
        metrics_df.to_csv(metrics_path, encoding='utf-8')

        # 
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Molecular Generation Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Generated Molecules: {num_molecules}\n")
            f.write(f"Complexity Level: {complexity_level}\n\n")

            f.write("Evaluation Metrics Comparison:\n")
            f.write("-" * 40 + "\n")
            for metric in ['Validity', 'Uniqueness', 'Novelty', 'Core Retention']:
                your_score = your_metrics[metric]
                baseline_score = baseline_metrics[metric]
                f.write(f"{metric}: Your Model={your_score:.3f}, Baseline Model={baseline_score:.3f}\n")

            f.write(f"\nConclusion:\n")
            f.write("-" * 20 + "\n")

            # 
            your_wins = 0
            total_metrics = 4

            for metric in ['Validity', 'Uniqueness', 'Core Retention']:
                if your_metrics[metric] > baseline_metrics[metric]:
                    your_wins += 1

            # ()
            if your_metrics['Novelty'] < baseline_metrics['Novelty']:
                your_wins += 1

            if your_wins > total_metrics / 2:
                f.write(" Your model outperforms the baseline model in most metrics!\n")
            elif your_wins == total_metrics / 2:
                f.write(" Your model performs comparably to the baseline model.\n")
            else:
                f.write(" Your model has room for improvement in some metrics.\n")

        print(f" ！: {output_dir}")
        print(f" :")
        print(f"   - : {results_path}")
        print(f"   - : {metrics_path}")
        print(f"   - : {report_path}")
        print(f"   - t-SNE: {tsne_path}")
        print(f"   - : {comparison_path}")

    except Exception as e:
        print(f" : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
