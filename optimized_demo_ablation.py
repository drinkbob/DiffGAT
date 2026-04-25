#!/usr/bin/env python3
"""
 - Property

:
1. Property
2. Property
3. 
4. C

:
python optimized_demo_ablation.py

WARNING:
This script contains synthetic/random data generation for demo visualization.
Do not use its outputs as manuscript-grade experimental evidence.
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
from sklearn.decomposition import PCA

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
    'xtick.minor.size': 2,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class OptimizedDemoAblationVisualizer:
    """"""

    def __init__(self):
        # 
        self.colors = {
            'DiffGAT (Full)': '#2E86AB',  # 
            '- Ring': '#F18F01',  # 
            '- Diversity': '#A23B72',  # 
            '- Ring & - Diversity': '#6C757D'  # 
        }

        # ()
        self.demo_metrics = {
            'DiffGAT (Full)': {
                'Validity': 0.95,
                'Uniqueness': 0.85,
                'Core Retention': 0.92,
                'Diversity': 0.78
            },
            '- Ring': {
                'Validity': 0.94,
                'Uniqueness': 0.65,
                'Core Retention': 0.88,
                'Diversity': 0.45
            },
            '- Diversity': {
                'Validity': 0.93,
                'Uniqueness': 0.45,
                'Core Retention': 0.91,
                'Diversity': 0.25
            },
            '- Ring & - Diversity': {
                'Validity': 0.92,
                'Uniqueness': 0.35,
                'Core Retention': 0.82,
                'Diversity': 0.15
            }
        }

        # Property - 
        self.demo_properties = {
            'DiffGAT (Full)': {
                'MW': np.random.normal(320, 50, 100),  # 
                'LogP': np.random.normal(3.2, 1.0, 100),  # LogP
                'HBD': np.random.poisson(3.5, 100),  # 
                'HBA': np.random.poisson(5.8, 100),  # 
                'TPSA': np.random.normal(85, 15, 100),  # 
                'RotBonds': np.random.poisson(4.2, 100)  # 
            },
            '- Ring': {
                'MW': np.random.normal(280, 40, 100),  # 
                'LogP': np.random.normal(2.8, 0.8, 100),  # LogP
                'HBD': np.random.poisson(2.8, 100),  # 
                'HBA': np.random.poisson(4.5, 100),  # 
                'TPSA': np.random.normal(75, 12, 100),  # TPSA
                'RotBonds': np.random.poisson(3.5, 100)  # 
            },
            '- Diversity': {
                'MW': np.random.normal(300, 45, 100),  # 
                'LogP': np.random.normal(3.0, 0.9, 100),  # LogP
                'HBD': np.random.poisson(3.0, 100),  # 
                'HBA': np.random.poisson(5.0, 100),  # 
                'TPSA': np.random.normal(80, 13, 100),  # TPSA
                'RotBonds': np.random.poisson(3.8, 100)  # 
            },
            '- Ring & - Diversity': {
                'MW': np.random.normal(260, 35, 100),  # 
                'LogP': np.random.normal(2.5, 0.7, 100),  # LogP
                'HBD': np.random.poisson(2.2, 100),  # 
                'HBA': np.random.poisson(3.8, 100),  # 
                'TPSA': np.random.normal(70, 10, 100),  # TPSA
                'RotBonds': np.random.poisson(3.0, 100)  # 
            }
        }

        # PCA - 
        self.demo_pca_data = {
            'DiffGAT (Full)': {
                'x': np.random.normal(0, 1.2, 80),  # 
                'y': np.random.normal(0, 1.2, 80)
            },
            '- Ring': {
                'x': np.random.normal(-1.5, 0.8, 80),  # 
                'y': np.random.normal(1.0, 0.8, 80)  # 
            },
            '- Diversity': {
                'x': np.random.normal(1.5, 0.6, 80),  # 
                'y': np.random.normal(-1.0, 0.6, 80)  # 
            },
            '- Ring & - Diversity': {
                'x': np.random.normal(0, 0.4, 80),  # 
                'y': np.random.normal(0, 0.4, 80)  # 
            }
        }

    def create_optimized_integrated_plot(self, save_path: str):
        """"""
        fig = plt.figure(figsize=(20, 16))

        # 1: 
        ax1 = plt.subplot(3, 3, 1)
        metrics = ['Validity', 'Uniqueness', 'Core Retention', 'Diversity']
        x = np.arange(len(metrics))
        width = 0.18

        # 4
        model_names = list(self.colors.keys())
        for i, model in enumerate(model_names):
            values = [self.demo_metrics[model].get(m, 0) for m in metrics]
            color = self.colors[model]
            bars = ax1.bar(x + i * width, values, width, label=model, color=color, alpha=0.8, edgecolor='black',
                           linewidth=0.5)

            # 
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax1.set_xlabel('Evaluation Metrics', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('(A) Ablation Study: Metrics Comparison', fontweight='bold', pad=15)
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right', fontsize=8)

        # 2: 
        ax2 = plt.subplot(3, 3, 2)

        # 
        box_data = []
        box_labels = []
        box_colors = []

        for metric in ['Validity', 'Uniqueness', 'Core Retention', 'Diversity']:
            for model in self.colors.keys():
                value = self.demo_metrics[model].get(metric, 0)
                # 
                if value > 0:
                    np.random.seed(42)
                    values = np.random.normal(value, value * 0.03, 30)
                    values = np.clip(values, 0, 1)
                else:
                    values = [0] * 30

                box_data.append(values)
                box_labels.append(f'{metric}\n{model}')
                box_colors.append(self.colors[model])

        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('(B) Metrics Distribution Across Variants', fontweight='bold', pad=15)
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

        # 3:  (Property)
        ax3 = plt.subplot(3, 3, 3)

        mw_data = []
        mw_labels = []
        mw_colors = []

        for model in self.colors.keys():
            values = self.demo_properties[model]['MW']
            mw_data.append(values)
            mw_labels.append(model)
            mw_colors.append(self.colors[model])

        bp3 = ax3.boxplot(mw_data, labels=mw_labels, patch_artist=True)
        for patch, color in zip(bp3['boxes'], mw_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_ylabel('Molecular Weight (Da)', fontweight='bold')
        ax3.set_title('(C) Molecular Weight Distribution', fontweight='bold', pad=15)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, axis='y', alpha=0.3, linestyle='--')

        # 4: LogP (Property)
        ax4 = plt.subplot(3, 3, 4)

        logp_data = []
        logp_labels = []
        logp_colors = []

        for model in self.colors.keys():
            values = self.demo_properties[model]['LogP']
            logp_data.append(values)
            logp_labels.append(model)
            logp_colors.append(self.colors[model])

        bp4 = ax4.boxplot(logp_data, labels=logp_labels, patch_artist=True)
        for patch, color in zip(bp4['boxes'], logp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax4.set_ylabel('LogP', fontweight='bold')
        ax4.set_title('(D) LogP Distribution', fontweight='bold', pad=15)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, axis='y', alpha=0.3, linestyle='--')

        # 5:  (Property)
        ax5 = plt.subplot(3, 3, 5)

        hb_data = []
        hb_labels = []
        hb_colors = []

        for model in self.colors.keys():
            hbd_values = self.demo_properties[model]['HBD']
            hba_values = self.demo_properties[model]['HBA']
            hb_data.append(hbd_values)
            hb_data.append(hba_values)
            hb_labels.append(f'HBD\n{model}')
            hb_labels.append(f'HBA\n{model}')
            hb_colors.append(self.colors[model])
            hb_colors.append(self.colors[model])

        bp5 = ax5.boxplot(hb_data, labels=hb_labels, patch_artist=True)
        for patch, color in zip(bp5['boxes'], hb_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax5.set_ylabel('Count', fontweight='bold')
        ax5.set_title('(E) Hydrogen Bond Donors & Acceptors', fontweight='bold', pad=15)
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, axis='y', alpha=0.3, linestyle='--')

        # 6: TPSA
        ax6 = plt.subplot(3, 3, 6)

        tpsa_data = []
        tpsa_labels = []
        tpsa_colors = []

        for model in self.colors.keys():
            values = self.demo_properties[model]['TPSA']
            tpsa_data.append(values)
            tpsa_labels.append(model)
            tpsa_colors.append(self.colors[model])

        bp6 = ax6.boxplot(tpsa_data, labels=tpsa_labels, patch_artist=True)
        for patch, color in zip(bp6['boxes'], tpsa_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax6.set_ylabel('TPSA (Å²)', fontweight='bold')
        ax6.set_title('(F) Topological Polar Surface Area', fontweight='bold', pad=15)
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, axis='y', alpha=0.3, linestyle='--')

        # 7: Property ()
        ax7 = plt.subplot(3, 3, 7)

        composite_data = []
        composite_labels = []
        composite_colors = []

        for model in self.colors.keys():
            props = self.demo_properties[model]

            # Property0-1
            mw_norm = (props['MW'] - 200) / (400 - 200)  # 200-400
            mw_norm = np.clip(mw_norm, 0, 1)

            logp_norm = (props['LogP'] - 1) / (5 - 1)  # 1-5
            logp_norm = np.clip(logp_norm, 0, 1)

            hbd_norm = props['HBD'] / 8  # 0-8
            hbd_norm = np.clip(hbd_norm, 0, 1)

            hba_norm = props['HBA'] / 10  # 0-10
            hba_norm = np.clip(hba_norm, 0, 1)

            tpsa_norm = (props['TPSA'] - 50) / (120 - 50)  # 50-120
            tpsa_norm = np.clip(tpsa_norm, 0, 1)

            rotbonds_norm = props['RotBonds'] / 8  # 0-8
            rotbonds_norm = np.clip(rotbonds_norm, 0, 1)

            # Property
            composite_score = (mw_norm + logp_norm + hbd_norm + hba_norm + tpsa_norm + rotbonds_norm) / 6

            composite_data.append(composite_score)
            composite_labels.append(model)
            composite_colors.append(self.colors[model])

        bp7 = ax7.boxplot(composite_data, labels=composite_labels, patch_artist=True)
        for patch, color in zip(bp7['boxes'], composite_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax7.set_ylabel('Composite Property Score', fontweight='bold')
        ax7.set_title('(G) Composite Molecular Properties', fontweight='bold', pad=15)
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax7.set_ylim(0, 1)

        # 8: 
        ax8 = plt.subplot(3, 3, 8)

        rotbonds_data = []
        rotbonds_labels = []
        rotbonds_colors = []

        for model in self.colors.keys():
            values = self.demo_properties[model]['RotBonds']
            rotbonds_data.append(values)
            rotbonds_labels.append(model)
            rotbonds_colors.append(self.colors[model])

        bp8 = ax8.boxplot(rotbonds_data, labels=rotbonds_labels, patch_artist=True)
        for patch, color in zip(bp8['boxes'], rotbonds_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax8.set_ylabel('Rotatable Bonds', fontweight='bold')
        ax8.set_title('(H) Rotatable Bonds Distribution', fontweight='bold', pad=15)
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, axis='y', alpha=0.3, linestyle='--')

        # 9: (PCA)
        ax9 = plt.subplot(3, 3, 9)

        # PCA - 
        for model, color in self.colors.items():
            x_coords = self.demo_pca_data[model]['x']
            y_coords = self.demo_pca_data[model]['y']
            # 
            if model == 'DiffGAT (Full)':
                alpha = 0.8
                s = 30
            elif model == '- Ring & - Diversity':
                alpha = 0.6
                s = 20
            else:
                alpha = 0.7
                s = 25
            ax9.scatter(x_coords, y_coords, c=color, label=model, alpha=alpha, s=s, edgecolors='white', linewidth=0.5)

        ax9.set_xlabel('PCA Component 1', fontweight='bold')
        ax9.set_ylabel('PCA Component 2', fontweight='bold')
        ax9.set_title('(I) Molecular Space Distribution (PCA)', fontweight='bold', pad=15)
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3, linestyle='--')

        #  - 
        fig.suptitle('Optimized Demo Ablation Study: Comprehensive Component Analysis of DiffGAT Model',
                     fontsize=18, fontweight='bold', y=0.98)

        # ,
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])  # 
        plt.subplots_adjust(hspace=0.4, wspace=0.3)  # 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" : {save_path}")

        return fig

def main():
    """Main function"""
    print("[WARNING] This script generates synthetic demo data (random distributions).")
    print("[WARNING] For explicit naming, prefer running optimized_demo_ablation_synthetic.py.")
    print(f" ")

    # 
    visualizer = OptimizedDemoAblationVisualizer()

    # 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'optimized_demo_ablation_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # 
    save_path = os.path.join(output_dir, 'Optimized_Demo_Ablation_Study_Comprehensive.png')
    visualizer.create_optimized_integrated_plot(save_path)

    # 
    metrics_df = pd.DataFrame(visualizer.demo_metrics).T
    metrics_df.to_csv(os.path.join(output_dir, 'optimized_demo_ablation_metrics.csv'))

    # Property
    with open(os.path.join(output_dir, 'optimized_molecular_properties.json'), 'w') as f:
        json_properties = {}
        for model, props in visualizer.demo_properties.items():
            json_properties[model] = {k: v.tolist() for k, v in props.items()}
        json.dump(json_properties, f, indent=2)

    # 
    with open(os.path.join(output_dir, 'optimized_ablation_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n')
        f.write('=' * 60 + '\n\n')
        f.write('\n\n')
        f.write(':\n')
        f.write('1. Property\n')
        f.write('2. Property\n')
        f.write('3. \n')
        f.write('4. C\n\n')

        f.write(':\n')
        for model, metrics in visualizer.demo_metrics.items():
            f.write(f'  {model}:\n')
            for metric, value in metrics.items():
                f.write(f'    {metric}: {value:.3f}\n')
            f.write('\n')

        f.write('Property:\n')
        for model, props in visualizer.demo_properties.items():
            f.write(f'  {model}:\n')
            for prop, values in props.items():
                f.write(
                    f'    {prop}: ={np.mean(values):.2f}, ={np.std(values):.2f}, =[{np.min(values):.2f}, {np.max(values):.2f}]\n')
            f.write('\n')

        f.write(':\n')
        f.write('1. DiffGAT (Full): ,\n')
        f.write('2. - Ring: Cyclic substituent generator,\n')
        f.write('3. - Diversity: Diversity control,\n')
        f.write('4. - Ring & - Diversity: ,\n')
        f.write('\n:\n')
        f.write('-  (Property)\n')
        f.write('- LogP (Property)\n')
        f.write('- HBDHBA (Property)\n')
        f.write('- TPSA\n')
        f.write('- Property\n')

    print(f" !")
    print(f"   : {output_dir}")
    print(f"   : Optimized_Demo_Ablation_Study_Comprehensive.png")
    print(f"   ")

if __name__ == '__main__':
    main()