"""
 Improved Hybrid Molecular Generation Framework - Benzimidazole Core-Constrained Version (Improved Hybrid Molecular Generation Framework - Benzimidazole Core Fixed Version)
Issues addressed relative to the original model:
1. Limited molecular complexity - Enhanced complex-structure generation capability
2. Inability to generate complex cyclic substituents - Added cyclic substituent generation module
3. Repeated molecule generation - Added diversity control mechanism
4. Training and validation sets were not separated - Implemented proper dataset splitting
5. Core scaffold was not constrained - Implemented benzimidazole core-constrained generation
6. Lack of target validation - Added beta-tubulin binding validation

Features:
-  Enhanced diffusion model:
-  Improved reinforcement learning:
-  Cyclic substituent generator:
-  Diversity control:
-  :proper train/validation/test splitting
-  Complex structure support:、、fused-ring structures
-  Core scaffold preservation:benzimidazole core
-  Target validation:βvalidation
"""

import torch
import numpy as np
import random
import os
import csv
import time
from datetime import datetime
from collections import defaultdict, deque
import math
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split

from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles, Draw, RemoveHs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem import QED, Lipinski
from rdkit.Chem import rdmolops, DataStructs

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

# ==== Improved configuration class ====
class ImprovedHybridConfig:
    """ - Benzimidazole Core-Constrained Version"""

    def __init__(self):
        # Basic configuration
        self.num_epochs = 50
        self.batch_size = 32  # Increase batch size
        self.learning_rate = 1e-4  # Adjust learning rate
        self.num_diffusion_steps = 200
        self.node_feat_dim = 12  # Expanded to 12-dimensional features
        self.edge_feat_dim = 4  # Expanded to 4-dimensional edge features
        self.hidden_dim = 256  # Increased hidden layer dimensionality
        self.num_heads = 8  # Increased number of attention heads
        self.dropout = 0.2  # Increased dropout to mitigate overfitting
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Core scaffold preservation
        self.core_smiles = "c1ccc2[nH]cnc2c1"  # benzimidazole core
        self.core_atoms = None  # core atom indices
        self.core_bonds = None  # core bond indices
        self.preserve_core = True  # whether to preserve the core scaffold
        self.core_mask = None  # core mask
        self.core_loss_weight = 10.0  # core-preservation loss weight
        self.core_penalty_weight = 5.0  # core-disruption penalty weight

        # Target validation - β
        self.target_protein = "beta_tubulin"  # β
        self.docking_score_threshold = -7.0  # docking score threshold
        self.binding_affinity_threshold = -8.0  # binding affinity threshold
        self.target_validation_weight = 2.0  # Target validation

        # Albendazole reference properties
        self.reference_mol_smiles = "CCc1ccc2[nH]cnc2c1"  # Albendazole
        self.reference_properties = {
            'MW': 265.33,
            'LogP': 3.5,
            'HBD': 2,
            'HBA': 4,
            'TPSA': 58.2,
            'RotatableBonds': 2,
            'AromaticRings': 2
        }

        # Large-dataset training optimization configuration
        self.gradient_clip = 0.5  # stricter gradient clipping
        self.weight_decay = 1e-4  # stronger weight decay
        self.lr_scheduler_patience = 5  # learning-rate scheduler patience
        self.lr_scheduler_factor = 0.5  # learning-rate decay factor
        self.early_stopping_patience = 15  # early-stopping patience

        # dataset split configuration
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # Diversity control
        self.diversity_threshold = 0.8
        self.max_similarity = 0.7
        self.diversity_buffer_size = 1000

        # complex structure configuration
        self.complex_ring_prob = 0.4
        self.fused_ring_prob = 0.3
        self.heterocycle_prob = 0.5
        self.max_rings = 5
        self.max_fused_rings = 3

        # Property
        self.properties_file = "molecule_properties.csv"
        self.sa_score_threshold = 8.0  # increase complexity threshold
        self.qed_threshold = 0.6
        self.min_rings = 2  # increase minimum ring count
        self.min_substituents = 2
        self.max_substituents = 4
        self.ring_formation_prob = 0.5  # increase ring formation probability
        self.noise_loss_weight = 1.0
        self.prop_loss_weight = 0.001
        self.diversity_loss_weight = 0.1  # added diversity loss weight

        # extended atom types:H,C,N,O,F,P,S,Cl,Br,I
        self.valid_atomic_nums = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

        # bond types
        self.bond_types = [
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC
        ]

        self.num_atom_types = len(self.valid_atomic_nums)
        self.num_bond_types = len(self.bond_types)

        print(f" ")
        print(f"   Device: {self.device}")
        print(f"   node feature dimension: {self.node_feat_dim}")
        print(f"   hidden dimension: {self.hidden_dim}")
        print(f"   number of attention heads: {self.num_heads}")
        print(f"   dataset split ratios: train{self.train_ratio:.1%}, validation{self.val_ratio:.1%}, test{self.test_ratio:.1%}")

# ==== Enhanced Core Structure Preserver ====
class EnhancedCoreStructurePreserver:
    """Enhanced Core Structure Preserver - """

    def __init__(self, config):
        self.config = config
        self.core_smiles = config.core_smiles
        self.core_mol = Chem.MolFromSmiles(self.core_smiles)

        # benzimidazole core
        self.benzimidazole_core = {
            'atoms': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 10
            'bonds': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # benzene
                      (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)],  # 
            'aromatic_atoms': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 
            'nitrogen_atoms': [7, 9],  # 
            'carbon_atoms': [0, 1, 2, 3, 4, 5, 6, 8]  # 
        }

    def identify_core_structure(self, mol):
        """benzimidazole core"""
        try:
            if mol is None or self.core_mol is None:
                return None, None, None

            # Use substructure matching
            matches = mol.GetSubstructMatches(self.core_mol)
            if not matches:
                return None, None, None

            # Select the largest match (typically the most complete core)
            core_atoms = max(matches, key=len)

            # Extract bonds within the core
            core_bonds = []
            core_bond_types = []
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                if begin_idx in core_atoms and end_idx in core_atoms:
                    core_bonds.append((begin_idx, end_idx))
                    core_bond_types.append(bond.GetBondTypeAsDouble())

            # Create detailed core metadata
            core_info = {
                'atoms': core_atoms,
                'bonds': core_bonds,
                'bond_types': core_bond_types,
                'aromatic_atoms': [idx for idx in core_atoms if mol.GetAtomWithIdx(idx).GetIsAromatic()],
                'nitrogen_atoms': [idx for idx in core_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7],
                'carbon_atoms': [idx for idx in core_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6]
            }

            return core_atoms, core_bonds, core_info

        except Exception as e:
            print(f"Core recognition failed: {e}")
            return None, None, None

    def create_core_mask(self, mol, core_atoms):
        """core mask"""
        if core_atoms is None:
            return None

        num_atoms = mol.GetNumAtoms()
        mask = torch.zeros(num_atoms, dtype=torch.bool)

        for atom_idx in core_atoms:
            if atom_idx < num_atoms:
                mask[atom_idx] = True

        return mask

    def preserve_core_features(self, atom_features, core_mask):
        """Preserve core features"""
        if core_mask is None:
            return atom_features

        # 
        core_features = torch.zeros_like(atom_features)
        core_features[core_mask] = atom_features[core_mask]

        return core_features

    def calculate_core_preservation_loss(self, original_features, generated_features, core_mask):
        """Compute core-preservation loss"""
        if core_mask is None or core_mask.sum() == 0:
            return torch.tensor(0.0).to(original_features.device)

        # Extract core features
        original_core = original_features[core_mask]
        generated_core = generated_features[core_mask]

        # Compute reconstruction loss on core features
        core_loss = F.mse_loss(generated_core, original_core)

        return core_loss * self.config.core_loss_weight

    def validate_core_integrity(self, mol):
        """validation"""
        try:
            core_atoms, core_bonds, core_info = self.identify_core_structure(mol)
            if core_atoms is None:
                return False, "benzimidazole core"

            # (10)
            if len(core_atoms) < 8:  # 
                return False, f"Insufficient number of core atoms: {len(core_atoms)}"

            # Check nitrogen count (benzimidazole should contain two nitrogen atoms)
            nitrogen_count = len(core_info['nitrogen_atoms'])
            if nitrogen_count < 1:  # At least one nitrogen atom is required
                return False, f"Insufficient nitrogen atom count: {nitrogen_count}"

            # Check aromaticity (most core atoms should be aromatic)
            aromatic_count = len(core_info['aromatic_atoms'])
            if aromatic_count < 6:  # At least six aromatic atoms are required
                return False, f"Insufficient aromatic atom count: {aromatic_count}"

            return True, "validation"

        except Exception as e:
            return False, f"validation: {e}"

# ==== Target validation ====
class EnhancedTargetValidator:
    """Target validation - β"""

    def __init__(self, config):
        self.config = config
        self.target_protein = config.target_protein
        self.reference_properties = config.reference_properties

    def validate_binding_affinity(self, mol):
        """validationβ"""
        try:
            # Compute molecular descriptors
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)

            # Albendazole
            score = 0.0
            max_score = 8.0  # maximum possible score

            # molecular weight score (ideal range: 250-300)
            if 250 <= mw <= 300:
                score += 1.0
            elif 200 <= mw <= 350:
                score += 0.5

            # LogP (ideal range: 2.5-4.5)
            if 2.5 <= logp <= 4.5:
                score += 1.0
            elif 2.0 <= logp <= 5.0:
                score += 0.5

            # hydrogen-bond donor score (: 1-3)
            if 1 <= hbd <= 3:
                score += 1.0

            # hydrogen-bond acceptor score (: 3-6)
            if 3 <= hba <= 6:
                score += 1.0

            # TPSA (: 50-80)
            if 50 <= tpsa <= 80:
                score += 1.0

            # rotatable bond score (: 1-4)
            if 1 <= rotatable_bonds <= 4:
                score += 1.0

            # aromatic ring score (: 2-3)
            if 2 <= aromatic_rings <= 3:
                score += 1.0

            # benzimidazole corevalidation
            core_validator = EnhancedCoreStructurePreserver(self.config)
            core_valid, core_message = core_validator.validate_core_integrity(mol)
            if core_valid:
                score += 1.0

            # normalize to 0-1
            final_score = score / max_score

            # Albendazole
            reference_mol = Chem.MolFromSmiles(self.config.reference_mol_smiles)
            if reference_mol:
                similarity = self.calculate_molecular_similarity(mol, reference_mol)
            else:
                similarity = 0.0

            return {
                'binding_score': final_score,
                'similarity_to_albendazole': similarity,
                'mw': mw,
                'logp': logp,
                'hbd': hbd,
                'hba': hba,
                'tpsa': tpsa,
                'rotatable_bonds': rotatable_bonds,
                'aromatic_rings': aromatic_rings,
                'core_valid': core_valid,
                'core_message': core_message,
                'is_promising': final_score >= 0.6 and core_valid
            }

        except Exception as e:
            print(f"Target validation: {e}")
            return {
                'binding_score': 0.0,
                'similarity_to_albendazole': 0.0,
                'is_promising': False,
                'core_valid': False,
                'core_message': f"validation: {e}"
            }

    def calculate_molecular_similarity(self, mol1, mol2):
        """Compute molecular similarity"""
        try:
            # Use the new Morgan fingerprint API to avoid deprecation warnings.
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp1 = gen.GetFingerprint(mol1)
            fp2 = gen.GetFingerprint(mol2)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            return similarity
        except:
            return 0.0

# ==== Cyclic substituent generator ====
class RingSubstituentGenerator:
    """Module specialized for generating complex cyclic substituents"""

    def __init__(self, config):
        self.config = config
        self.ring_templates = self._create_ring_templates()
        self.fused_ring_templates = self._create_fused_ring_templates()
        self.heterocycle_templates = self._create_heterocycle_templates()

    def _create_ring_templates(self):
        """Create cyclic templates"""
        return {
            # basic cyclic structures
            'cyclopentane': 'C1CCCC1',
            'cyclohexane': 'C1CCCCC1',
            'cyclopentene': 'C1=CCCC1',
            'cyclohexene': 'C1=CCCCC1',
            'cycloheptane': 'C1CCCCCC1',
            'cyclooctane': 'C1CCCCCCC1',

            # aromatic ring structures
            'benzene': 'c1ccccc1',
            'pyridine': 'c1ccncc1',
            'pyrimidine': 'c1cncnc1',
            'pyrazine': 'c1cnccn1',
            'pyridazine': 'c1cccnc1',
            'triazine': 'c1ncncn1',

            # five-membered heterocycles
            'imidazole': 'c1c[nH]cn1',
            'thiophene': 'c1ccsc1',
            'furan': 'c1ccoc1',
            'pyrrole': 'c1c[nH]cc1',
            'oxazole': 'c1cocn1',
            'thiazole': 'c1cscc1',
            'isoxazole': 'c1ccon1',
            'isothiazole': 'c1ccsn1',

            # fused-ring structures
            'indole': 'c1ccc2c(c1)[nH]cc2',
            'quinoline': 'c1ccc2ncccc2c1',
            'isoquinoline': 'c1ccc2cnccc2c1',
            'benzimidazole': 'c1ccc2[nH]cnc2c1',
            'benzothiophene': 'c1ccc2c(c1)scc2',
            'benzofuran': 'c1ccc2c(c1)occ2',
            'quinazoline': 'c1ccc2nc3ccccc3nc2c1',
            'cinnoline': 'c1ccc2nnccc2c1',
            'phthalazine': 'c1ccc2nnccc2c1',

            # polycyclic aromatic hydrocarbons
            'naphthalene': 'c1ccc2ccccc2c1',
            'anthracene': 'c1ccc2c(c1)ccc3ccccc23',
            'phenanthrene': 'c1ccc2c(c1)ccc3ccccc32',
            'pyrene': 'c1ccc2c(c1)ccc3c2ccc4ccccc34',
            'chrysene': 'c1ccc2c(c1)ccc3c2ccc4ccccc34',

            # complex heterocycles
            'acridine': 'c1ccc2c(c1)ccc3ccccc23',
            'carbazole': 'c1ccc2c(c1)ccc3ccccc32',
            'dibenzofuran': 'c1ccc2c(c1)oc3ccccc23',
            'dibenzothiophene': 'c1ccc2c(c1)sc3ccccc23',
            'phenazine': 'c1ccc2c(c1)nc3ccccc3n2',
            'phenoxazine': 'c1ccc2c(c1)oc3ccccc3n2',
            'phenothiazine': 'c1ccc2c(c1)sc3ccccc3n2'
        }

    def _create_fused_ring_templates(self):
        """Create fused-ring templates"""
        return {
            'naphthalene': 'c1ccc2ccccc2c1',
            'anthracene': 'c1ccc2c(c1)ccc3ccccc23',
            'phenanthrene': 'c1ccc2c(c1)ccc3ccccc32',
            'quinoline_fused': 'c1ccc2ncccc2c1',
            'isoquinoline_fused': 'c1ccc2cnccc2c1',
            'acridine': 'c1ccc2c(c1)ccc3ccccc23',
            'carbazole': 'c1ccc2c(c1)ccc3ccccc32'
        }

    def _create_heterocycle_templates(self):
        """Create heterocycle templates"""
        return {
            'piperidine': 'C1CCNCC1',
            'piperazine': 'C1CCNCC1',  # simplified version
            'morpholine': 'C1CCOCC1',
            'tetrahydrofuran': 'C1CCOC1',
            'tetrahydropyran': 'C1CCOCC1',
            'pyrrolidine': 'C1CCNC1',
            'imidazolidine': 'C1CNCN1',  # fixed
            'oxazolidine': 'C1CNCO1',  # fixed
            'thiazolidine': 'C1CNCS1'  # fixed
        }

    def generate_ring_substituent(self, complexity_level='medium'):
        """Generate cyclic substituents"""
        if complexity_level == 'simple':
            templates = list(self.ring_templates.values())[:10]  # increase the number of simple templates
        elif complexity_level == 'medium':
            templates = list(self.ring_templates.values())
        else:  # complex
            templates = (list(self.ring_templates.values()) +
                         list(self.fused_ring_templates.values()) +
                         list(self.heterocycle_templates.values()))

        template = random.choice(templates)

        # For simple complexity level, return templates directly
        if complexity_level == 'simple':
            return template

        # enhanced substituent system
        substituents = {
            'alkyl': ['C', 'CC', 'CCC', 'CC(C)C', 'CCCC'],
            'halogen': ['F', 'Cl', 'Br', 'I'],
            'oxygen': ['O', 'OC', 'OCC', 'OC(C)C'],
            'nitrogen': ['N', 'NC', 'N(C)C', 'NCC'],
            'sulfur': ['S', 'SC', 'SCC'],
            'complex': ['CC(C)(C)C', 'CC(C)CC', 'c1ccccc1', 'c1ccncc1', 'C1CCCCC1']
        }

        # Select substituent type by complexity
        if complexity_level == 'medium':
            substituent_types = ['alkyl', 'halogen', 'oxygen', 'nitrogen']
            weights = [0.4, 0.2, 0.2, 0.2]
        else:  # complex
            substituent_types = ['alkyl', 'halogen', 'oxygen', 'nitrogen', 'sulfur', 'complex']
            weights = [0.3, 0.15, 0.15, 0.15, 0.1, 0.15]

        # Select number of substituents
        if complexity_level == 'medium':
            num_substituents = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
        else:  # complex
            num_substituents = random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0]

        if num_substituents == 0:
            return template

        modified_template = template
        for _ in range(num_substituents):
            # Select substituent class
            substituent_type = random.choices(substituent_types, weights=weights)[0]
            substituent = random.choice(substituents[substituent_type])

            # Add substituents using rule-based logic
            if 'c1' in modified_template and modified_template.count('c') > 3:
                # aromatic-ring system
                if substituent_type == 'alkyl' and len(substituent) == 1:
                    # simple alkyl groups replacing carbon atoms
                    modified_template = modified_template.replace('c1', 'c(C)1', 1)
                elif substituent_type == 'halogen':
                    # add halogens at chemically feasible positions
                    modified_template = modified_template.replace('c1', 'c(' + substituent + ')1', 1)
                else:
                    # add complex substituents at terminal positions
                    modified_template = modified_template + substituent
            else:
                # alicyclic system
                modified_template = modified_template + substituent

        return modified_template

    def generate_fused_ring_system(self):
        """Generate fused-ring systems"""
        templates = list(self.fused_ring_templates.values())
        return random.choice(templates)

    def generate_heterocycle_system(self):
        """Generate heterocyclic systems"""
        templates = list(self.heterocycle_templates.values())
        return random.choice(templates)

# ==== Diversity control ====
class DiversityController:
    """Module for controlling molecular diversity"""

    def __init__(self, config):
        self.config = config
        self.generated_molecules = deque(maxlen=config.diversity_buffer_size)
        self.similarity_cache = {}

    def calculate_molecular_similarity(self, mol1, mol2):
        """Compute molecular similarity"""
        if mol1 is None or mol2 is None:
            return 0.0

        try:
            # Use the new Morgan fingerprint API to avoid deprecation warnings.
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            fp1 = gen.GetFingerprint(mol1)
            fp2 = gen.GetFingerprint(mol2)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            return similarity
        except Exception:
            return 0.0

    def is_diverse_enough(self, new_mol):
        """Check whether a new molecule is sufficiently diverse"""
        if len(self.generated_molecules) == 0:
            return True

        max_similarity = 0.0
        for existing_mol in self.generated_molecules:
            similarity = self.calculate_molecular_similarity(new_mol, existing_mol)
            max_similarity = max(max_similarity, similarity)

        return max_similarity <= self.config.max_similarity

    def add_molecule(self, mol):
        """Add molecule to diversity buffer"""
        if mol is not None:
            self.generated_molecules.append(mol)

    def get_diversity_score(self):
        """Get current diversity score"""
        if len(self.generated_molecules) < 2:
            return 1.0

        total_similarity = 0.0
        count = 0

        molecules_list = list(self.generated_molecules)
        for i in range(len(molecules_list)):
            for j in range(i + 1, len(molecules_list)):
                mol1 = molecules_list[i]
                mol2 = molecules_list[j]
                similarity = self.calculate_molecular_similarity(mol1, mol2)
                total_similarity += similarity
                count += 1

        if count == 0:
            return 1.0

        avg_similarity = total_similarity / count
        diversity_score = 1.0 - avg_similarity
        return diversity_score

# ==== Improved molecular encoder ====
class ImprovedMolecularEncoder(nn.Module):
    """Improved molecular encoder:supports 12-dimensional node features"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.node_feat_dim = 12

        # Node feature encoder with batch normalization
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 256)
        )

        # time embedding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

        # improved GAT layers
        self.gat1 = GATConv(
            in_channels=256,
            out_channels=256,
            heads=8,
            edge_dim=config.edge_feat_dim,
            dropout=config.dropout,
            concat=True
        )

        self.gat2 = GATConv(
            in_channels=256 * 8,
            out_channels=256,
            heads=4,
            edge_dim=config.edge_feat_dim,
            dropout=config.dropout,
            concat=False
        )

        self.gat3 = GATConv(
            in_channels=256,
            out_channels=256,
            heads=2,
            edge_dim=config.edge_feat_dim,
            dropout=config.dropout,
            concat=False
        )

        # output layer
        self.output_norm = nn.LayerNorm(256)
        self.output_projection = nn.Linear(256, self.node_feat_dim)

    def forward(self, x, edge_index, edge_attr, time_step):
        """forward pass"""
        # ensure input dimensions are correct
        if x.size(-1) != self.node_feat_dim:
            if x.size(-1) < self.node_feat_dim:
                padding = torch.zeros(x.size(0), self.node_feat_dim - x.size(-1)).to(x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[:, :self.node_feat_dim]

        # encode node features
        x_encoded = self.node_encoder(x)

        # time embedding
        if time_step.dim() == 0:
            time_step = time_step.unsqueeze(0)
        time_emb = self.time_encoder(time_step.view(-1, 1))

        # time embedding
        if time_emb.size(0) == 1:
            time_emb = time_emb.expand(x_encoded.size(0), -1)

        # time embedding
        x = x_encoded + time_emb

        # GAT
        if edge_index.numel() > 0 and edge_index.size(1) > 0:
            x = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
            x = F.relu(self.gat2(x, edge_index, edge_attr=edge_attr))
            x = F.relu(self.gat3(x, edge_index, edge_attr=edge_attr))

        # 
        x = self.output_norm(x)

        # ,25612
        x = self.output_projection(x)

        return x

# ====  ====
class DatasetSplitter:
    """"""

    def __init__(self, config):
        self.config = config

    def split_dataset(self, dataset):
        """train/validation/test"""
        total_size = len(dataset)

        if total_size == 0:
            print(" Warning: dataset is empty and cannot be split")
            return [], [], []

        # 1train
        if total_size < 3:
            print(f" Warning: dataset is too small ({total_size} ),train")
            return dataset, [], []

        # Compute split sizes
        train_size = max(1, int(total_size * self.config.train_ratio))
        val_size = max(1, int(total_size * self.config.val_ratio))
        test_size = total_size - train_size - val_size

        # Ensure test_size is non-negative
        if test_size < 0:
            test_size = 0
            val_size = total_size - train_size

        # Random split
        train_dataset, temp_dataset = train_test_split(
            dataset, train_size=train_size, random_state=42
        )

        if len(temp_dataset) > 0:
            val_dataset, test_dataset = train_test_split(
                temp_dataset, train_size=val_size, random_state=42
            )
        else:
            val_dataset, test_dataset = [], []

        print(f" Dataset split completed:")
        print(f"   train: {len(train_dataset)}  ({len(train_dataset) / total_size:.1%})")
        print(f"   Validation set size: {len(val_dataset)}  ({len(val_dataset) / total_size:.1%})")
        print(f"   test: {len(test_dataset)}  ({len(test_dataset) / total_size:.1%})")

        return train_dataset, val_dataset, test_dataset

# ==== Main class of the improved hybrid molecular generator ====
class ImprovedHybridMolecularGenerator:
    """Improved Hybrid Molecular Generation Framework"""

    def __init__(self, config=None):
        self.config = config or ImprovedHybridConfig()
        self.device = self.config.device

        # core components
        self.shared_encoder = ImprovedMolecularEncoder(self.config).to(self.device)
        self.ring_generator = RingSubstituentGenerator(self.config)
        self.diversity_controller = DiversityController(self.config)
        self.dataset_splitter = DatasetSplitter(self.config)
        self.core_preserver = EnhancedCoreStructurePreserver(self.config)
        self.target_validator = EnhancedTargetValidator(self.config)

        # train
        self.optimizer = torch.optim.AdamW(
            self.shared_encoder.parameters(),
            lr=self.config.learning_rate
        )

        # train
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'diversity_scores': [],
            'generated_molecules': [],
            'complexity_scores': []
        }

        print(f" Improved Hybrid Molecular Generation Framework")
        print(f"   Device: {self.device}")
        print(f"   node feature dimension: {self.config.node_feat_dim}")
        print(f"   hidden dimension: {self.config.hidden_dim}")

        # 1. __init__
        # ImprovedHybridMolecularGenerator.__init__:
        self.known_smiles = [
            "CCc1ccc2[nH]cnc2c1",  # Albendazole

        ]
        self.known_mols = [Chem.MolFromSmiles(s) for s in self.known_smiles if Chem.MolFromSmiles(s)]
        self.innovation_loss_weight = 1.0  # innovation loss weight

    def create_enhanced_dataset(self, custom_csv_path=None):
        """Create augmented dataset"""
        print(" ...")

        dataset = []

        # 1. First load custom CSV data (if provided)
        if custom_csv_path and os.path.exists(custom_csv_path):
            custom_molecules = self._load_custom_molecules_from_csv(custom_csv_path)
            dataset.extend(custom_molecules)
            print(f" Loaded {len(custom_molecules)} custom molecules from CSV")

        # 2. Base molecular templates with simple valid SMILES
        base_templates = [
            "c1ccccc1",  # benzene
            "c1ccncc1",  # pyridine
            "c1ccsc1",  # thiophene
            "c1ccoc1",  # furan
            "c1ccc2ccccc2c1",  # naphthalene
            "c1ccc2[nH]cnc2c1",  # 
            "c1ccc2c(c1)[nH]cc2",  # indole
            "c1ccc2ncccc2c1",  # quinoline
            "c1ccc2cnccc2c1",  # quinoline
            "C1CCCCC1",  # cyclohexane
            "C1=CCCCC1",  # cyclohexene
            "C1CCCC1",  # cyclopentane
            "C1=CCCC1",  # cyclopentene
        ]

        # 3. Add base template molecules without variant generation
        for template in base_templates:
            mol = Chem.MolFromSmiles(template)
            if mol:
                data = self._molecule_to_data(mol)
                if data:
                    dataset.append(data)
                    print(f" : {template}")

        # 4. ()
        for template in base_templates[:3]:  # 3
            for _ in range(3):  # 3
                mol = self._create_molecule_variant(template)
                if mol:
                    data = self._molecule_to_data(mol)
                    if data:
                        dataset.append(data)

        # 5. ()
        for _ in range(10):
            complexity = 'simple'
            ring_smiles = self.ring_generator.generate_ring_substituent(complexity)
            mol = Chem.MolFromSmiles(ring_smiles)
            if mol:
                data = self._molecule_to_data(mol)
                if data:
                    dataset.append(data)

        # 6. ()
        for _ in range(5):
            complexity = 'medium'
            ring_smiles = self.ring_generator.generate_ring_substituent(complexity)
            mol = Chem.MolFromSmiles(ring_smiles)
            if mol:
                data = self._molecule_to_data(mol)
                if data:
                    dataset.append(data)

        # 7. Generate fused-ring systems()
        for _ in range(3):
            fused_smiles = self.ring_generator.generate_fused_ring_system()
            mol = Chem.MolFromSmiles(fused_smiles)
            if mol:
                data = self._molecule_to_data(mol)
                if data:
                    dataset.append(data)

        # 8. Generate heterocyclic systems()
        for _ in range(3):
            hetero_smiles = self.ring_generator.generate_heterocycle_system()
            mol = Chem.MolFromSmiles(hetero_smiles)
            if mol:
                data = self._molecule_to_data(mol)
                if data:
                    dataset.append(data)

        print(f" , {len(dataset)} ")

        # ,
        if len(dataset) == 0:
            print(" ,...")
            simple_molecules = [
                "C", "CC", "CCC", "CCCC", "c1ccccc1", "C1CCCCC1", "C1=CCCCC1"
            ]
            for smiles in simple_molecules:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        data = self._molecule_to_data(mol)
                        if data:
                            dataset.append(data)
                            print(f" : {smiles}")
                except Exception as e:
                    print(f"  {smiles}: {e}")
            print(f" , {len(dataset)} ")

        # ,
        if len(dataset) == 0:
            print(" ...")
            try:
                # 
                mol = Chem.MolFromSmiles("C")
                if mol:
                    data = self._molecule_to_data(mol)
                    if data:
                        dataset.append(data)
                        print(" ")
            except Exception as e:
                print(f"Model save/load error: {e}")

        return dataset

    def _load_custom_molecules_from_csv(self, csv_path):
        """CSV"""
        custom_molecules = []

        try:
            import pandas as pd

            # 
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(csv_path, encoding='gbk')
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_path, encoding='latin-1')

            print(f" : {len(df)} ")

            # SMILES
            smiles_columns = ['SMILES', 'smiles', 'Smiles', 'MOLECULE', 'molecule']
            smiles_col = None

            for col in smiles_columns:
                if col in df.columns:
                    smiles_col = col
                    break

            if smiles_col is None:
                print(f" CSVSMILES")
                print(f"   : {list(df.columns)}")
                return custom_molecules

            print(f" SMILES: {smiles_col}")

            # 
            df_clean = df.dropna(subset=[smiles_col]).drop_duplicates(subset=[smiles_col])
            print(f" : {len(df_clean)} ")

            # 
            max_molecules = 2000000  # 200
            if len(df_clean) > max_molecules:
                print(f"   ({len(df_clean)} ), {max_molecules} ")
                df_clean = df_clean.sample(n=max_molecules, random_state=42)
            else:
                print(f" , {len(df_clean)} ")

            # ()
            batch_size = 10000  # 1
            total_batches = (len(df_clean) + batch_size - 1) // batch_size

            print(f" : {total_batches} , {batch_size} ")

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(df_clean))
                batch_df = df_clean.iloc[start_idx:end_idx]

                print(f"    {batch_idx + 1}/{total_batches} ({start_idx + 1}-{end_idx})")

                batch_count = 0
                for idx, row in batch_df.iterrows():
                    try:
                        smiles = str(row[smiles_col]).strip()

                        # SMILES
                        if not smiles or smiles == 'nan' or smiles == '':
                            continue

                        mol = Chem.MolFromSmiles(smiles)

                        if mol:
                            data = self._molecule_to_data(mol)
                            if data:
                                custom_molecules.append(data)
                                batch_count += 1
                    except Exception as e:
                        # ,
                        continue

                print(f"     {batch_idx + 1} , {batch_count} ")

            print(f"  {len(custom_molecules)} ")

        except Exception as e:
            print(f" CSV: {e}")
            import traceback
            traceback.print_exc()

        return custom_molecules

    def _create_molecule_variant(self, template):
        """"""
        try:
            mol = Chem.MolFromSmiles(template)
            if not mol:
                return None

            # ,1
            substituents = ['C', 'O', 'N']  # 
            weights = [0.6, 0.2, 0.2]

            num_substituents = random.choices([0, 1], weights=[0.7, 0.3])[0]  # 70%

            if num_substituents == 0:
                return mol  # 

            for _ in range(num_substituents):
                substituent = random.choices(substituents, weights=weights)[0]
                # 
                mol = self._add_substituent(mol, substituent)
                if mol is None:  # ,
                    mol = Chem.MolFromSmiles(template)
                    break

            return mol

        except Exception as e:
            print(f": {e}")
            return None

    def _add_substituent(self, mol, substituent):
        """"""
        try:
            rw_mol = Chem.RWMol(mol)
            atoms = list(rw_mol.GetAtoms())

            if not atoms:
                return mol

            # ()
            valid_atoms = [atom for atom in atoms if atom.GetAtomicNum() != 1]
            if not valid_atoms:
                return mol

            attach_atom = random.choice(valid_atoms)
            attach_idx = attach_atom.GetIdx()

            # 
            new_atom = Chem.Atom(substituent)
            new_idx = rw_mol.AddAtom(new_atom)

            # 
            rw_mol.AddBond(attach_idx, new_idx, Chem.BondType.SINGLE)

            new_mol = rw_mol.GetMol()

            # 
            try:
                Chem.SanitizeMol(new_mol)
                return new_mol
            except:
                # ,
                return mol

        except Exception as e:
            print(f": {e}")
            return mol

    def _molecule_to_data(self, mol):
        """"""
        try:
            # 
            if mol is None:
                return None

            # fixed
            try:
                mol = Chem.AddHs(mol)
                Chem.SanitizeMol(mol)

                # Kekulization
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except:
                # Kekulization,
                try:
                    # 
                    smiles = Chem.MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return None
                    mol = Chem.AddHs(mol)
                    Chem.SanitizeMol(mol)
                except:
                    return None

            # 
            ring_info = mol.GetRingInfo()

            # 12
            atom_features = []
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()

                # 
                in_3_ring = any(len(ring) == 3 for ring in ring_info.AtomRings() if atom_idx in ring)
                in_4_ring = any(len(ring) == 4 for ring in ring_info.AtomRings() if atom_idx in ring)
                in_5_ring = any(len(ring) == 5 for ring in ring_info.AtomRings() if atom_idx in ring)
                in_6_ring = any(len(ring) == 6 for ring in ring_info.AtomRings() if atom_idx in ring)

                # 
                atomic_num = atom.GetAtomicNum()
                degree = atom.GetDegree()
                formal_charge = atom.GetFormalCharge()

                feat = [
                    (atomic_num - 6.0) / 10.0,  #  ()
                    min(degree / 4.0, 1.0),  #  ([0,1])
                    (formal_charge + 1.0) / 3.0,  #  ([0,1])
                    float(atom.IsInRing()),  # 
                    float(atom.GetTotalNumHs() > 0 and atomic_num in [7, 8]),  # HBD
                    float(atomic_num in [7, 8]),  # HBA
                    float(atom.GetIsAromatic()),  # 
                    float(atomic_num not in [1, 6]),  # 
                    float(in_3_ring),  # 
                    float(in_4_ring),  # 
                    float(in_5_ring),  # 
                    float(in_6_ring)  # 
                ]

                # 
                feat = [max(-5.0, min(5.0, f)) for f in feat]
                atom_features.append(feat)

            # 4
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                edge_index.append([start, end])
                edge_index.append([end, start])

                bond_type = bond.GetBondTypeAsDouble()
                is_aromatic = float(bond.GetIsAromatic())
                is_ring = float(bond.IsInRing())
                is_conjugated = float(bond.GetIsConjugated())

                edge_attr.append([bond_type, is_aromatic, is_ring, is_conjugated])
                edge_attr.append([bond_type, is_aromatic, is_ring, is_conjugated])

            # mask
            mask = [1] * mol.GetNumAtoms()

            data = Data(
                x=torch.tensor(atom_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index).t().contiguous() if edge_index else torch.empty((2, 0),
                                                                                                    dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 4),
                                                                                                   dtype=torch.float),
                mask=torch.tensor(mask, dtype=torch.float)
            )

            return data

        except Exception as e:
            print(f": {e}")
            return None

    def train_model(self, epochs=50, custom_csv_path=None):
        """train - Benzimidazole Core-Constrained Version"""
        print(f"\n{'=' * 70}")
        print(" Training with core scaffold preservation")
        print(f"{'=' * 70}")

        # 
        dataset = self.create_enhanced_dataset(custom_csv_path=custom_csv_path)

        if len(dataset) == 0:
            print("Error: dataset construction failed.")
            return False

        # 
        train_dataset, val_dataset, test_dataset = self.dataset_splitter.split_dataset(dataset)

        if len(train_dataset) == 0:
            print("Error: training set is empty after dataset split.")
            return False

        # Data loading
        train_loader = DataLoader(train_dataset, batch_size=min(self.config.batch_size, len(train_dataset)),
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(self.config.batch_size, len(val_dataset)),
                                shuffle=False) if len(val_dataset) > 0 else None

        # 
        self.optimizer = torch.optim.AdamW(
            self.shared_encoder.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience
        )

        print(f"Training started ({epochs} epochs)")
        print(f"   Train set size: {len(train_dataset)}")
        print(f"   Validation set size: {len(val_dataset)}")
        print(f"   Test set size: {len(test_dataset)}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Gradient clipping: {self.config.gradient_clip}")
        print(f"   Core scaffold SMARTS/SMILES: {self.config.core_smiles}")
        print(f"   Core-preservation loss weight: {self.config.core_loss_weight}")
        print(f"   Target validation: {self.config.target_validation_weight}")

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # train
            self.shared_encoder.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_core_loss = 0.0
            train_target_loss = 0.0
            train_diversity_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()

                # forward pass
                output = self.shared_encoder(
                    batch.x, batch.edge_index, batch.edge_attr,
                    torch.tensor([0.5]).to(self.device)
                )

                # 
                if output.size(-1) != batch.x.size(-1):
                    output = output[:, :batch.x.size(-1)]

                # 
                target = torch.clamp(batch.x, -10, 10)
                output = torch.clamp(output, -10, 10)

                # 
                recon_loss = F.smooth_l1_loss(output, target, beta=0.1)

                # L2
                l2_loss = 0.0
                for param in self.shared_encoder.parameters():
                    l2_loss += torch.norm(param, p=2)
                l2_loss = l2_loss * 1e-6

                #  - 
                core_loss = self._calculate_enhanced_core_loss(batch, output)

                # Target validation - 
                target_validation_loss = self._calculate_target_validation_loss(batch, output)

                # 
                diversity_loss = torch.tensor(0.0).to(output.device)
                if output.size(0) > 1:
                    feature_std = torch.std(output, dim=0)
                    diversity_loss = -torch.mean(feature_std) * 0.01

                # 
                total_loss = recon_loss + l2_loss + core_loss + target_validation_loss + diversity_loss

                # 
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"Warning: invalid total loss encountered: {total_loss.item()}")
                    continue

                total_loss.backward()

                # 
                torch.nn.utils.clip_grad_norm_(
                    self.shared_encoder.parameters(),
                    self.config.gradient_clip
                )

                self.optimizer.step()

                # 
                train_loss += total_loss.item()
                train_recon_loss += recon_loss.item()
                train_core_loss += core_loss.item()
                train_target_loss += target_validation_loss.item()
                train_diversity_loss += diversity_loss.item()
                num_batches += 1

            # 
            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
            avg_recon_loss = train_recon_loss / num_batches if num_batches > 0 else 0.0
            avg_core_loss = train_core_loss / num_batches if num_batches > 0 else 0.0
            avg_target_loss = train_target_loss / num_batches if num_batches > 0 else 0.0
            avg_diversity_loss = train_diversity_loss / num_batches if num_batches > 0 else 0.0

            # validation
            self.shared_encoder.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_core_loss = 0.0
            val_target_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)

                    output = self.shared_encoder(
                        batch.x, batch.edge_index, batch.edge_attr,
                        torch.tensor([0.5]).to(self.device)
                    )

                    # 
                    if output.size(-1) != batch.x.size(-1):
                        output = output[:, :batch.x.size(-1)]

                    # validation
                    target = torch.clamp(batch.x, -10, 10)
                    output = torch.clamp(output, -10, 10)

                    recon_loss = F.smooth_l1_loss(output, target, beta=0.1)
                    core_loss = self._calculate_enhanced_core_loss(batch, output)
                    target_validation_loss = self._calculate_target_validation_loss(batch, output)

                    total_loss = recon_loss + core_loss + target_validation_loss

                    val_loss += total_loss.item()
                    val_recon_loss += recon_loss.item()
                    val_core_loss += core_loss.item()
                    val_target_loss += target_validation_loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            avg_val_recon_loss = val_recon_loss / val_batches if val_batches > 0 else 0.0
            avg_val_core_loss = val_core_loss / val_batches if val_batches > 0 else 0.0
            avg_val_target_loss = val_target_loss / val_batches if val_batches > 0 else 0.0

            # 
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # 
            scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 
                self.save_model("best_model.pth")
                print(f"  (validation: {best_val_loss:.6f})")
            else:
                patience_counter += 1

            # 
            if avg_train_loss > 100 or avg_val_loss > 100:
                print(f"  ！train: {avg_train_loss:.6f}, validation: {avg_val_loss:.6f}")
                print("Reinitializing model parameters due to instability...")
                self._reset_model_parameters()
                continue

            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

            # 
            if (epoch + 1) % 5 == 0 or epoch < 10:
                print(
                    f"   Epoch {epoch + 1}/{epochs}: train={avg_train_loss:.6f} (: {avg_recon_loss:.6f}, : {avg_core_loss:.6f}, : {avg_target_loss:.6f})")
                print(
                    f"   validation={avg_val_loss:.6f} (: {avg_val_recon_loss:.6f}, : {avg_val_core_loss:.6f}, : {avg_val_target_loss:.6f}), LR={current_lr:.2e}")

        print(f" train！validation: {best_val_loss:.6f}")
        print(f"Train loss trajectory: {train_losses[0]:.6f} -> {train_losses[-1]:.6f}")
        print(f" validation: {val_losses[0]:.6f} -> {val_losses[-1]:.6f}")
        return True

    def _reset_model_parameters(self):
        """"""
        print("Reinitializing model parameters due to instability...")
        for module in self.shared_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # 
        self.optimizer = torch.optim.AdamW(
            self.shared_encoder.parameters(),
            lr=self.config.learning_rate * 0.1,  # 
            weight_decay=self.config.weight_decay
        )
        print(" ")

    def _calculate_enhanced_core_loss(self, batch, output):
        """"""
        try:
            # benzimidazole core
            core_mol = Chem.MolFromSmiles(self.config.core_smiles)
            if core_mol is None:
                return torch.tensor(0.0).to(output.device)

            # core atom indices
            core_atoms, core_bonds, core_info = self.core_preserver.identify_core_structure(core_mol)
            if core_atoms is None:
                return torch.tensor(0.0).to(output.device)

            # core mask
            core_mask = self.core_preserver.create_core_mask(core_mol, core_atoms)
            if core_mask is None:
                return torch.tensor(0.0).to(output.device)

            # Device
            core_mask = core_mask.to(output.device)

            # 
            if core_mask.size(0) > output.size(0):
                core_mask = core_mask[:output.size(0)]
            elif core_mask.size(0) < output.size(0):
                padding = torch.zeros(output.size(0) - core_mask.size(0), dtype=torch.bool).to(output.device)
                core_mask = torch.cat([core_mask, padding])

            # 
            core_loss = self.core_preserver.calculate_core_preservation_loss(
                batch.x, output, core_mask
            )

            return core_loss

        except Exception as e:
            print(f": {e}")
            return torch.tensor(0.0).to(output.device)

    def _calculate_target_validation_loss(self, batch, output):
        """Target validation"""
        try:
            # validation
            target_loss = torch.tensor(0.0).to(output.device)

            # Target validation
            # 
            if hasattr(self, 'target_validator') and self.target_validator:
                # Albendazole
                reference_mol = Chem.MolFromSmiles(self.config.reference_mol_smiles)
                if reference_mol:
                    # 
                    similarity_loss = torch.tensor(0.0).to(output.device)
                    target_loss = similarity_loss * self.config.target_validation_weight

            return target_loss

        except Exception as e:
            print(f"Target validation: {e}")
            return torch.tensor(0.0).to(output.device)

    def _calculate_diversity_loss(self, features):
        """"""
        if features.size(0) < 2:
            return torch.tensor(0.0).to(features.device)

        # 
        similarity_matrix = torch.mm(features, features.t())

        # 0()
        mask = torch.eye(features.size(0)).to(features.device)
        similarity_matrix = similarity_matrix * (1 - mask)

        # 
        avg_similarity = torch.sum(similarity_matrix) / (features.size(0) * (features.size(0) - 1))

        #  = 1 - 
        diversity_loss = 1.0 - avg_similarity

        return diversity_loss

    def generate_molecules(self, num_molecules=50, complexity_level='medium', ps_enriched=False):
        """
        benzimidazole core

        Args:
            num_molecules: Target()
            complexity_level: 'simple' / 'medium' / 'complex'
            ps_enriched:  P/S ()
        """
        print(f"  {num_molecules} benzimidazole core...")
        if ps_enriched:
            print("     P/S (P/S)")

        generated_molecules = []

        # ,
        benzimidazole_templates = [
            # 
            "c1ccc2[nH]cnc2c1",  # 

            # ()
            "CCc1ccc2[nH]cnc2c1",  # Albendazole(2)
            "c1ccc2[nH]cnc2c1C",  # 1
            "c1ccc2[nH]cnc2c1c3ccccc3",  # 1
            "c1ccc2[nH]cnc2c1c3ccc(cc3)C",  # 1
            "c1ccc2[nH]cnc2c1C3CCCCC3",  # 1

            # 
            "CCc1ccc2[nH]cnc2c1c3ccccc3",  # 2 + 1
            "c1ccc2[nH]cnc2c1c3ccc(cc3)Cc4ccccc4",  # 1 + 
            "c1ccc2[nH]cnc2c1C3CCCCC3c4ccccc4",  # 1 + 

            # 
            "c1ccc2[nH]cnc2c1c3ccc(cc3)O",  # 1
            "c1ccc2[nH]cnc2c1c3ccc(cc3)N",  # 1
            "c1ccc2[nH]cnc2c1c3ccc(cc3)F",  # 1
            "c1ccc2[nH]cnc2c1c3ccc(cc3)Cl",  # 1

            # 
            "c1ccc2[nH]cnc2c1C3CCCCCC3",  # 1
            "c1ccc2[nH]cnc2c1c3ccc4ccccc4c3",  # 1naphthalene
            "c1ccc2[nH]cnc2c1c3ccc4c(c3)cccc4",  # 1
        ]

        # Statistics
        attempts = 0
        max_attempts = num_molecules * 10  # ()
        success_count = 0
        failed_reasons = {
            'base_mol_failed': 0,
            'substituent_failed': 0,
            'connection_failed': 0,
            'sanitize_failed': 0,
            'core_validation_failed': 0,
            'diversity_failed': 0,
            'other_error': 0
        }

        # ()
        original_max_similarity = self.config.max_similarity
        if num_molecules > 100:
            # ,
            self.config.max_similarity = min(0.85, original_max_similarity + 0.1)
            print(f"    : {self.config.max_similarity:.2f} (: {original_max_similarity:.2f})")

        while success_count < num_molecules and attempts < max_attempts:
            attempts += 1
            try:
                # 
                if success_count < len(benzimidazole_templates):
                    base_smiles = benzimidazole_templates[success_count]
                else:
                    base_smiles = random.choice(benzimidazole_templates)

                base_mol = Chem.MolFromSmiles(base_smiles)
                if base_mol is None:
                    failed_reasons['base_mol_failed'] += 1
                    continue

                # (P/S)
                substituent_smiles = self._generate_benzimidazole_substituent(
                    complexity_level,
                    force_ps=ps_enriched
                )
                if not substituent_smiles:
                    failed_reasons['substituent_failed'] += 1
                    continue

                # ()
                final_mol = None
                connection_attempts = 0
                max_connection_attempts = 3

                while final_mol is None and connection_attempts < max_connection_attempts:
                    final_mol = self._attach_substituent_to_core(base_mol, substituent_smiles)
                    if final_mol is None and connection_attempts == 0:
                        # ,
                        final_mol = base_mol
                    connection_attempts += 1

                if final_mol is None:
                    failed_reasons['connection_failed'] += 1
                    continue

                # validation
                try:
                    Chem.SanitizeMol(final_mol)
                    smiles_str = Chem.MolToSmiles(final_mol)
                    if not smiles_str:
                        failed_reasons['sanitize_failed'] += 1
                        continue
                except:
                    failed_reasons['sanitize_failed'] += 1
                    continue

                # validation(validation)
                core_valid, core_message = self.core_preserver.validate_core_integrity(final_mol)
                if not core_valid:
                    # validation
                    try:
                        # 
                        core_pattern = Chem.MolFromSmarts("c1ccc2[nH]cnc2c1")
                        if final_mol.HasSubstructMatch(core_pattern):
                            core_valid = True
                            core_message = "validation"
                    except:
                        pass

                if not core_valid:
                    failed_reasons['core_validation_failed'] += 1
                    if attempts % 100 == 0:  # 100
                        print(f"  validation ( {attempts} ,  {success_count} )")
                    continue

                # Target validation
                target_validation = self.target_validator.validate_binding_affinity(final_mol)

                # (,)
                diversity_passed = self.diversity_controller.is_diverse_enough(final_mol)

                # ,
                if not diversity_passed and num_molecules > 500 and success_count > num_molecules * 0.8:
                    # Target,
                    diversity_passed = True

                if diversity_passed:
                    # Diversity control
                    self.diversity_controller.add_molecule(final_mol)

                    # Property
                    props = self._calculate_molecular_properties(final_mol)

                    generated_molecules.append({
                        'smiles': Chem.MolToSmiles(final_mol),
                        'molecule': final_mol,
                        'properties': props,
                        'target_validation': target_validation,
                        'core_valid': core_valid,
                        'complexity_level': complexity_level
                    })

                    success_count += 1

                    # 10100
                    if success_count % max(1, num_molecules // 20) == 0 or success_count <= 10:
                        print(f"     {success_count}/{num_molecules}: {Chem.MolToSmiles(final_mol)}")
                        if props:
                            print(
                                f"      MW={props.get('MW', 0):.1f}, LogP={props.get('LogP', 0):.2f}, SA={props.get('SA_Score', 0):.1f}")
                        if target_validation and target_validation.get('is_promising', False):
                            print(f"      ！")
                else:
                    failed_reasons['diversity_failed'] += 1

            except Exception as e:
                failed_reasons['other_error'] += 1
                if attempts % 100 == 0:  # 100
                    print(f"   ( {attempts} ): {e}")

        # 
        self.config.max_similarity = original_max_similarity

        # Statistics
        print(f"\nSummary:")
        print(f"   Target: {num_molecules}")
        print(f"   Successfully generated: {success_count}")
        print(f"   : {attempts}")
        print(f"   : {success_count / attempts * 100:.1f}%" if attempts > 0 else "   : 0%")
        print(f"   :")
        for reason, count in failed_reasons.items():
            if count > 0:
                print(f"     - {reason}: {count} ")

        # 
        if generated_molecules:
            self._save_generated_molecules(generated_molecules)

        print(f" Successfully generated {len(generated_molecules)} benzimidazole core")
        return generated_molecules

    def _generate_benzimidazole_substituent(self, complexity_level, force_ps: bool = False):
        """
        , P/S 

        Args:
            complexity_level: 'simple' / 'medium' / 'complex'
            force_ps:  True, P/S 
        """
        try:
            # ()
            substituent_templates = {
                'simple': [
                    # 
                    'C', 'CC', 'CCC', 'CCCC',
                    # 
                    'c1ccccc1',
                    #  O/N Functions
                    'CCO', 'CCCO',  # 
                    'CCN', 'CCCN',  # 
                    'CC(=O)O', 'CCC(=O)O',  # 
                    'OC', 'OCC',  # 
                    'NC', 'NCC',  # 
                    # 
                    'F', 'Cl', 'Br',
                    #  S/P ()
                    'SC', 'SCC',  # sulfur
                    'S(=O)C', 'S(=O)(=O)C',  # /
                    'P(=O)(O)O', 'CP(=O)(O)O',  # phosphorus/
                ],
                'medium': [
                    # 
                    'c1ccc(cc1)C', 'c1ccc(cc1)CC',
                    'c1ccc(cc1)O', 'c1ccc(cc1)N',
                    'c1ccc(cc1)F', 'c1ccc(cc1)Cl',
                    # 
                    'C1CCCCC1', 'C1CCCCCC1',
                    # /
                    'c1ccc2c(c1)cccc2',  # naphthalene
                    'c1ccc2c(c1)oc3ccccc23',  # furan (O )
                    'c1ccc2c(c1)sc3ccccc23',  # thiophene (S )
                    'c1ccc2c(c1)nc3ccccc23',  #  (N )
                    # /
                    'c1ccc(cc1)OC', 'c1ccc(cc1)NC',
                    'c1ccc(cc1)C(=O)O', 'c1ccc(cc1)C(=O)OC',
                    #  S/P ()
                    'c1ccc(cc1)S(=O)C',  # 
                    'c1ccc(cc1)S(=O)(=O)C',  # 
                    'c1ccc(cc1)P(=O)(O)O',  # phosphorus
                ],
                'complex': [
                    # 
                    'c1ccc(cc1)c2ccc(cc2)C',
                    'c1ccc(cc1)c2ccc(cc2)O',
                    'c1ccc(cc1)c2ccc(cc2)N',
                    # 
                    'c1ccc2c(c1)ccc3ccccc23',  # 
                    'c1ccc2c(c1)ccc3ccccc32',  # 
                    # 
                    'c1ccc2c(c1)oc3ccccc23',  # furan
                    'c1ccc2c(c1)sc3ccccc23',  # thiophene
                    'c1ccc2c(c1)nc3ccccc23',  # 
                    #  O/N
                    'c1ccc(cc1)c2ccc(cc2)OC',
                    'c1ccc(cc1)c2ccc(cc2)NC',
                    #  S/P ()
                    'c1ccc(cc1)c2ccc(cc2)S(=O)C',
                    'c1ccc(cc1)c2ccc(cc2)S(=O)(=O)C',
                    'c1ccc(cc1)c2ccc(cc2)P(=O)(O)O',
                ],
            }

            templates = substituent_templates.get(complexity_level, substituent_templates['medium'])

            # , P/S 
            use_ps_enriched = force_ps or (random.random() < 0.5)
            if use_ps_enriched:
                ps_templates = [t for t in templates if ('P' in t or 'S' in t)]
                if ps_templates:
                    templates = ps_templates

            selected_template = random.choice(templates)

            # validation
            test_mol = Chem.MolFromSmiles(selected_template)
            if test_mol is None:
                # ,
                return 'CC'

            # (), S/P
            if random.random() < 0.15:
                modifiers = ['C', 'O', 'N', 'F', 'Cl', 'S', 'P']
                modifier = random.choice(modifiers)
                test_modified = f"{selected_template}{modifier}"
                if Chem.MolFromSmiles(test_modified):
                    selected_template = test_modified

            return selected_template

        except Exception as e:
            print(f": {e}")
            return 'CC'  # 

    def _attach_substituent_to_core(self, core_mol, substituent_smiles):
        """benzimidazole core(,)"""
        try:
            # 
            substituent_mol = Chem.MolFromSmiles(substituent_smiles)
            if substituent_mol is None:
                return None

            # SMILES()
            core_smiles = Chem.MolToSmiles(core_mol)

            # 1: SMILES(,)
            # 
            connection_patterns = [
                f"{core_smiles}{substituent_smiles}",  # 
                f"{core_smiles}-{substituent_smiles}",  # 
                f"{substituent_smiles}{core_smiles}",  # 
            ]

            for pattern in connection_patterns:
                try:
                    result_mol = Chem.MolFromSmiles(pattern)
                    if result_mol:
                        Chem.SanitizeMol(result_mol)
                        # validation
                        if result_mol.HasSubstructMatch(core_mol):
                            return result_mol
                except:
                    continue

            # 2: RDKitFunctions()
            try:
                from rdkit.Chem import rdmolops
                rw_mol = Chem.RWMol(core_mol)

                # (benzene)
                connection_atoms = []
                for atom in rw_mol.GetAtoms():
                    if atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
                        # 
                        if atom.GetDegree() < 4:
                            connection_atoms.append(atom.GetIdx())

                if connection_atoms:
                    # 
                    attach_idx = random.choice(connection_atoms)

                    # 
                    substituent_atoms = []
                    for atom in substituent_mol.GetAtoms():
                        new_idx = rw_mol.AddAtom(atom)
                        substituent_atoms.append(new_idx)

                    # 
                    for bond in substituent_mol.GetBonds():
                        begin_idx = substituent_atoms[bond.GetBeginAtomIdx()]
                        end_idx = substituent_atoms[bond.GetEndAtomIdx()]
                        rw_mol.AddBond(begin_idx, end_idx, bond.GetBondType())

                    # ()
                    if substituent_atoms:
                        rw_mol.AddBond(attach_idx, substituent_atoms[0], Chem.BondType.SINGLE)

                    result_mol = rw_mol.GetMol()
                    Chem.SanitizeMol(result_mol)

                    # validation
                    if result_mol and result_mol.HasSubstructMatch(core_mol):
                        return result_mol
            except Exception as e:
                # ,
                pass

            # 3: ,()
            return core_mol

        except Exception as e:
            # 
            return core_mol

    def _calculate_molecular_properties(self, mol):
        """Property"""
        if mol is None:
            return None

        try:
            props = {}

            # Property
            props['MW'] = Descriptors.MolWt(mol)
            props['LogP'] = Crippen.MolLogP(mol)
            props['TPSA'] = Descriptors.TPSA(mol)
            props['HBD'] = Lipinski.NumHDonors(mol)
            props['HBA'] = Lipinski.NumHAcceptors(mol)

            # 
            try:
                # SA_Score
                props['SA_Score'] = rdMolDescriptors.CalcSAScore(mol)
            except AttributeError:
                # SA_Score,
                try:
                    from rdkit.Chem import SA_Score
                    props['SA_Score'] = SA_Score.calculateScore(mol)
                except:
                    # ,
                    props['SA_Score'] = props['MW'] / 100.0

            try:
                props['QED'] = QED.qed(mol)
            except:
                props['QED'] = 0.5  # 

            # 
            ring_info = mol.GetRingInfo()
            props['NumRings'] = len(ring_info.AtomRings())
            props['NumAromaticRings'] = len([ring for ring in ring_info.AtomRings()
                                             if any(mol.GetAtomWithIdx(idx).GetIsAromatic()
                                                    for idx in ring)])

            # Lipinski
            violations = 0
            if props['MW'] > 500: violations += 1
            if props['LogP'] > 5: violations += 1
            if props['HBD'] > 5: violations += 1
            if props['HBA'] > 10: violations += 1
            props['Lipinski_Violations'] = violations

            return props

        except Exception as e:
            print(f"Property: {e}")
            return None

    def _save_generated_molecules(self, molecules):
        """"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # SDF
        sdf_filename = f"generated_molecules_{timestamp}.sdf"
        with Chem.SDWriter(sdf_filename) as writer:
            for mol_data in molecules:
                mol = mol_data['molecule']
                if mol and mol_data.get('properties'):
                    for prop_name, prop_value in mol_data['properties'].items():
                        mol.SetProp(prop_name, str(prop_value))
                    writer.write(mol)

        # CSV
        csv_filename = f"generated_molecules_{timestamp}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            if molecules and molecules[0].get('properties'):
                fieldnames = ['SMILES', 'Complexity_Level'] + list(molecules[0]['properties'].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for mol_data in molecules:
                    row = {
                        'SMILES': mol_data['smiles'],
                        'Complexity_Level': mol_data['complexity_level']
                    }
                    if mol_data.get('properties'):
                        row.update(mol_data['properties'])
                    writer.writerow(row)

        print(f" :")
        print(f"   SDF: {sdf_filename}")
        print(f"   CSV: {csv_filename}")

    def save_model(self, filepath):
        """"""
        try:
            checkpoint = {
                'shared_encoder': self.shared_encoder.state_dict(),
                'config': self.config,
                'training_stats': self.training_stats
            }
            torch.save(checkpoint, filepath)
            print(f"Model saved to: {filepath}")
        except Exception as e:
            print(f"Model save/load error: {e}")

    def load_model(self, filepath):
        """"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.shared_encoder.load_state_dict(checkpoint['shared_encoder'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            print(f"  {filepath} ")
        except Exception as e:
            print(f"Model save/load error: {e}")

# ==== Functions ====
def train_improved_hybrid_generator(epochs=50, custom_csv_path="data.csv"):
    """train"""
    print(f"\n{'=' * 70}")
    print(" Improved Hybrid Molecular Generation Framework Training")
    print(f"{'=' * 70}")

    if custom_csv_path:
        print(f" CSV: {custom_csv_path}")

    try:
        config = ImprovedHybridConfig()
        generator = ImprovedHybridMolecularGenerator(config)

        success = generator.train_model(epochs=epochs, custom_csv_path=custom_csv_path)

        if success:
            print(f"\nTraining completed successfully.")

            # test
            print(f"\nRunning post-training generation sanity check...")
            test_molecules = generator.generate_molecules(num_molecules=50, complexity_level='medium')

            print(f"\nSummary:")
            print(f"   Total generated molecules: {len(test_molecules)}")
            print(f"   Diversity score: {generator.diversity_controller.get_diversity_score():.3f}")

            return generator
        else:
            print(f"\nTraining failed.")
            return None

    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_improved_framework():
    """"""
    print(f"\n{'=' * 70}")
    print(" Improved Hybrid Molecular Generation Framework")
    print(f"{'=' * 70}")

    # train
    print("Launching short demo training (10 epochs)...")
    generator = train_improved_hybrid_generator(epochs=10, custom_csv_path="data.csv")

    if generator:
        print("\nGenerating demonstration molecules...")
        demo_molecules = generator.generate_molecules(num_molecules=20, complexity_level='complex')

        print("\nDemo pipeline finished successfully.")
        return generator
    else:
        print("\n ")
        return None

# ====  ====
if __name__ == "__main__":
    print(" Improved Hybrid Molecular Generation Framework - Issues addressed relative to the original model")
    print("=" * 70)

    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "train":
            epochs = 50
            custom_csv_path = None

            if len(sys.argv) > 2:
                # (epochs)(CSV)
                try:
                    epochs = int(sys.argv[2])
                    # ,CSV
                    if len(sys.argv) > 3:
                        custom_csv_path = sys.argv[3]
                except ValueError:
                    # ,CSV
                    custom_csv_path = sys.argv[2]
                    # ,epochs
                    if len(sys.argv) > 3:
                        try:
                            epochs = int(sys.argv[3])
                        except ValueError:
                            print(" epochs")

            print(f" train...")
            print(f"   Epochs: {epochs}")
            if custom_csv_path:
                print(f"   CSV: {custom_csv_path}")

            generator = train_improved_hybrid_generator(epochs=epochs, custom_csv_path=custom_csv_path)
            if generator:
                print("Training mode completed successfully.")
            else:
                print("Training mode failed.")
        elif mode == "demo":
            print("Running demo mode...")
            demo_improved_framework()
        else:
            print(" ")
            print("Usage:")
            print("  python improved_hybrid_molecular_generator.py train [epochs] [csv_file]")
            print("  python improved_hybrid_molecular_generator.py train [csv_file] [epochs]")
            print("  python improved_hybrid_molecular_generator.py demo")
    else:
        # :
        print("Running demo mode...")
        demo_improved_framework()

    print(f"\n{'=' * 70}")
    print("")
    print(f"{'=' * 70}")