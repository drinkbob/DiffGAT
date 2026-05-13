import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ========== 1.  ==========
from improved_hybrid_molecular_generator import ImprovedHybridMolecularGenerator
model = ImprovedHybridMolecularGenerator(...)
generated_smiles = model.generate(num_samples=1000)  # SMILES

# ========== 2. train ==========
train_smiles = [...]  # trainSMILES
baseline_smiles = [...]  # SMILES

# ========== 3.  ==========
def calc_validity(smiles_list):
    return np.mean([Chem.MolFromSmiles(smi) is not None for smi in smiles_list])

def calc_uniqueness(smiles_list):
    return len(set(smiles_list)) / len(smiles_list)

def calc_novelty(smiles_list, train_set):
    return len(set(smiles_list) - set(train_set)) / len(set(smiles_list))

def calc_core_retention(smiles_list, core_smarts):
    patt = Chem.MolFromSmarts(core_smarts)
    return np.mean([Chem.MolFromSmiles(smi) is not None and Chem.MolFromSmiles(smi).HasSubstructMatch(patt) for smi in smiles_list])

# ========== 4.  ==========
core_smarts = "clccc2[nH]cnc2c1"  # SMARTS
your_validity = calc_validity(generated_smiles)
your_uniqueness = calc_uniqueness(generated_smiles)
your_novelty = calc_novelty(generated_smiles, train_smiles)
your_core = calc_core_retention(generated_smiles, core_smarts)

baseline_validity = calc_validity(baseline_smiles)
baseline_uniqueness = calc_uniqueness(baseline_smiles)
baseline_novelty = calc_novelty(baseline_smiles, train_smiles)
baseline_core = calc_core_retention(baseline_smiles, core_smarts)

# ========== 5. t-SNE  ==========
def smiles_to_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))

def get_fps(smiles_list):
    fps = []
    for smi in smiles_list:
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
    return np.array(fps)

fps_train = get_fps(train_smiles)
fps_yours = get_fps(generated_smiles)
fps_baseline = get_fps(baseline_smiles)

all_fps = np.vstack([fps_train, fps_yours, fps_baseline])
tsne = TSNE(n_components=2, random_state=42)
all_emb = tsne.fit_transform(all_fps)

n1, n2 = len(fps_train), len(fps_yours)
plt.figure(figsize=(8,6))
plt.scatter(all_emb[:n1,0], all_emb[:n1,1], c='blue', label='Train', alpha=0.5, s=10)
plt.scatter(all_emb[n1:n1+n2,0], all_emb[n1:n1+n2,1], c='red', label='Your Model', alpha=0.5, s=10)
plt.scatter(all_emb[n1+n2:,0], all_emb[n1+n2:,1], c='green', label='Baseline', alpha=0.5, s=10)
plt.legend()
plt.title('t-SNE of Molecular Fingerprints')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# ========== 6.  ==========
labels = ['Validity', 'Uniqueness', 'Novelty', 'Core Retention']
your_scores = [your_validity, your_uniqueness, your_novelty, your_core]
baseline_scores = [baseline_validity, baseline_uniqueness, baseline_novelty, baseline_core]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, your_scores, width, label='Your Model')
rects2 = ax.bar(x + width/2, baseline_scores, width, label='Baseline')

ax.set_ylabel('Scores')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0, 1)
plt.show()
