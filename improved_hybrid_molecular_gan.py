#!/usr/bin/env python3
"""
improved_hybrid_molecular_gan.py
GAT + Property Prediction + GAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, Lipinski
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ==== GAT generator (consistent with the original model) ====
class ImprovedMolecularEncoder(nn.Module):
    def __init__(self, node_feat_dim=12, edge_feat_dim=4, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, time_step=None):
        if x.size(-1) != self.node_feat_dim:
            if x.size(-1) < self.node_feat_dim:
                padding = torch.zeros(x.size(0), self.node_feat_dim - x.size(-1)).to(x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[:, :self.node_feat_dim]
        x_encoded = self.node_encoder(x)
        x = self.output_norm(x_encoded)
        return x

# ==== Property Prediction ====
class PropertyPredictor(nn.Module):
    def __init__(self, input_dim=256, num_props=3):  # QED, LogP, MW
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_props)
        )
    def forward(self, x):
        return self.net(x)

# ==== Discriminator ====
class MoleculeDiscriminator(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# ==== Convert SMILES to molecular graph data ====
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except:
        pass
    atom_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        feat = [
            (atomic_num - 6.0) / 10.0,
            min(degree / 4.0, 1.0),
            (formal_charge + 1.0) / 3.0,
            float(atom.IsInRing()),
            float(atom.GetTotalNumHs() > 0 and atomic_num in [7, 8]),
            float(atomic_num in [7, 8]),
            float(atom.GetIsAromatic()),
            float(atomic_num not in [1, 6]),
            0, 0, 0, 0  # Extensible ring features
        ]
        atom_features.append(feat)
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
    if not atom_features:
        return None
    data = Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 4), dtype=torch.float)
    )
    return data

# ==== Property Prediction ====
def calc_properties(mol):
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        qed = QED.qed(mol)
        return [qed, logp, mw]
    except:
        return [0.5, 0.0, 0.0]

# ==== Data loading ====
def load_smiles_dataset(csv_path, smiles_col='SMILES', max_num=None):
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
    smiles_list = df[smiles_col].dropna().unique().tolist()
    if max_num:
        smiles_list = smiles_list[:max_num]
    graph_list = []
    prop_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        data = smiles_to_graph(smi)
        if data is None:
            continue
        props = calc_properties(mol)
        graph_list.append(data)
        prop_list.append(props)
    return graph_list, np.array(prop_list, dtype=np.float32)

# ==== GAN Training Workflow ====
class GANTrainer:
    def __init__(self, generator, discriminator, property_predictor, device, property_loss_weight=1.0):
        self.generator = generator
        self.discriminator = discriminator
        self.property_predictor = property_predictor
        self.device = device
        self.gen_optimizer = torch.optim.AdamW(
            list(generator.parameters()) + list(property_predictor.parameters()), lr=1e-4)
        self.disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()
        self.prop_criterion = nn.MSELoss()
        self.property_loss_weight = property_loss_weight

    def train(self, train_loader, real_property_tensor, epochs=30):
        for epoch in range(epochs):
            for i, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                # 1. Load dataTrain the discriminator
                self.disc_optimizer.zero_grad()
                real_x = self.generator(batch.x, batch.edge_index, batch.edge_attr)
                real_label = torch.ones(real_x.size(0), 1, device=self.device)
                real_pred = self.discriminator(real_x)
                real_loss = self.criterion(real_pred, real_label)
                # 
                noise = torch.randn_like(batch.x)
                fake_x = self.generator(noise, batch.edge_index, batch.edge_attr)
                fake_label = torch.zeros(fake_x.size(0), 1, device=self.device)
                fake_pred = self.discriminator(fake_x.detach())
                fake_loss = self.criterion(fake_pred, fake_label)
                disc_loss = real_loss + fake_loss
                disc_loss.backward()
                self.disc_optimizer.step()

                # 2. Train the generator
                self.gen_optimizer.zero_grad()
                fake_pred = self.discriminator(fake_x)
                gen_loss = self.criterion(fake_pred, real_label)
                # Property prediction loss
                prop_pred = self.property_predictor(fake_x)
                prop_loss = self.prop_criterion(prop_pred, real_property_tensor[:prop_pred.size(0)].to(self.device))
                total_gen_loss = gen_loss + prop_loss * self.property_loss_weight
                total_gen_loss.backward()
                self.gen_optimizer.step()

            print(f'Epoch {epoch+1}/{epochs} | D_loss: {disc_loss.item():.4f} | G_loss: {gen_loss.item():.4f} | Prop_loss: {prop_loss.item():.4f}')

# ==== Main entry point ====
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='SMILES dataset CSV file')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_num', type=int, default=10000, help='Maximum number of molecules to load')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1. Load data
    print('Loading SMILES data...')
    graph_list, prop_array = load_smiles_dataset(args.csv, max_num=args.max_num)
    print(f'Number of valid molecules: {len(graph_list)}')
    prop_tensor = torch.tensor(prop_array, dtype=torch.float)

    # 2. Data loading
    loader = DataLoader(graph_list, batch_size=args.batch_size, shuffle=True)

    # 3. 
    generator = ImprovedMolecularEncoder().to(device)
    property_predictor = PropertyPredictor(256, num_props=3).to(device)
    discriminator = MoleculeDiscriminator(256).to(device)

    # 4. Launch training
    trainer = GANTrainer(generator, discriminator, property_predictor, device, property_loss_weight=1.0)
    trainer.train(loader, prop_tensor, epochs=args.epochs)

    print('Training completed successfully.')