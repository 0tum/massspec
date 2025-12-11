"""
Generate BOTH 2D (Physicochemical) and 3D (Shape/Geometric) descriptors.
Reads from .npz file (containing 'smiles' array) instead of parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from tqdm import tqdm

def _disable_rdkit_logging() -> None:
    RDLogger.DisableLog("rdApp.*")

# -------------------------------------------------------------------------
# 1. 2D Descriptor Calculation
# -------------------------------------------------------------------------
def calc_2d_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute ~200 RDKit 2D descriptors."""
    vals = []
    for name, func in Descriptors.descList:
        try:
            v = func(mol)
            if not np.isfinite(v):
                v = 0.0
            vals.append(v)
        except:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)

# -------------------------------------------------------------------------
# 2. 3D Descriptor Calculation
# -------------------------------------------------------------------------
def calc_3d_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute ~900 RDKit 3D descriptors."""
    feats: List[float] = []
    
    # Shape & Scalars
    feats.append(Descriptors3D.PMI1(mol))
    feats.append(Descriptors3D.PMI2(mol))
    feats.append(Descriptors3D.PMI3(mol))
    feats.append(Descriptors3D.NPR1(mol))
    feats.append(Descriptors3D.NPR2(mol))
    feats.append(Descriptors3D.RadiusOfGyration(mol))
    feats.append(Descriptors3D.InertialShapeFactor(mol))
    feats.append(Descriptors3D.SpherocityIndex(mol))
    
    # Vectors
    feats.extend(rdMolDescriptors.CalcMORSE(mol))
    feats.extend(rdMolDescriptors.CalcRDF(mol))
    feats.extend(rdMolDescriptors.CalcWHIM(mol))
    feats.extend(rdMolDescriptors.CalcGETAWAY(mol))
    feats.extend(rdMolDescriptors.CalcAUTOCORR3D(mol))
    
    return np.array(feats, dtype=np.float32)

# -------------------------------------------------------------------------
# Dimensions Helper
# -------------------------------------------------------------------------
def get_dims() -> Tuple[int, int]:
    """Determine 2D and 3D descriptor dimensions using a dummy molecule."""
    mol = Chem.MolFromSmiles("CC") 
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    
    d2 = len(calc_2d_descriptors(mol))
    d3 = len(calc_3d_descriptors(mol))
    return d2, d3

# -------------------------------------------------------------------------
# Main Processing Logic
# -------------------------------------------------------------------------
def embed_and_calc(
    smiles: str,
    max_atoms: int,
    params: AllChem.EmbedParameters,
    max_embed_attempts: int,
    dim2: int,
    dim3: int,
    optimize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int, bool, str]:
    """Returns (desc2d, desc3d, atom_count, success, error_msg)."""
    
    empty2 = np.zeros(dim2, dtype=np.float32)
    empty3 = np.zeros(dim3, dtype=np.float32)

    # A. Mol Prep
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return empty2, empty3, 0, False, "Invalid SMILES"
        
        # 2D計算
        desc2d = calc_2d_descriptors(mol) 
        
        # 水素付加 & 原子数チェック
        mol = Chem.AddHs(mol) 
        n_atoms = mol.GetNumAtoms()
        if n_atoms > max_atoms:
            return desc2d, empty3, n_atoms, False, f"Too large ({n_atoms} > {max_atoms})"
            
    except Exception as e:
        return empty2, empty3, 0, False, f"Prep Error: {e}"

    # B. 3D Embedding
    original_seed = params.randomSeed
    res = -1
    try:
        for attempt in range(max_embed_attempts):
            params.randomSeed = original_seed + attempt
            res = AllChem.EmbedMolecule(mol, params)
            if res == 0: break
            mol.RemoveAllConformers()
    except Exception as e:
        return desc2d, empty3, 0, False, f"Embed Error: {e}"

    if res != 0:
        return desc2d, empty3, 0, False, "Embed Failed"

    # C. Optimization
    if optimize:
        try: AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except: pass

    # D. 3D Calc
    try:
        desc3d = calc_3d_descriptors(mol)
        if len(desc3d) != dim3:
            if len(desc3d) < dim3: desc3d = np.pad(desc3d, (0, dim3 - len(desc3d)))
            else: desc3d = desc3d[:dim3]
            
        return desc2d, desc3d, mol.GetNumAtoms(), True, ""
    except Exception as e:
        return desc2d, empty3, 0, False, f"3D Calc Error: {e}"

def process_smiles(
    smiles_list: Iterable[str],
    max_atoms: int,
    seed: int,
    max_embed_attempts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    
    print("Determining feature dimensions...")
    dim2, dim3 = get_dims()
    print(f"2D Descriptors: {dim2} dims")
    print(f"3D Descriptors: {dim3} dims")
    
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0 

    cache: Dict[str, Tuple[np.ndarray, np.ndarray, int, bool]] = {}

    n = len(smiles_list)
    arr_2d = np.zeros((n, dim2), dtype=np.float32)
    arr_3d = np.zeros((n, dim3), dtype=np.float32)
    atom_counts = np.zeros(n, dtype=np.int32)
    success = np.zeros(n, dtype=bool)

    for idx, smi in enumerate(tqdm(smiles_list, desc="Processing", unit="mol")):
        if smi in cache:
            d2, d3, count, ok = cache[smi]
        else:
            d2, d3, count, ok, err = embed_and_calc(smi, max_atoms, params, max_embed_attempts, dim2, dim3)
            cache[smi] = (d2, d3, count, ok)

        arr_2d[idx] = d2
        arr_3d[idx] = d3
        atom_counts[idx] = count
        success[idx] = ok

    return arr_2d, arr_3d, atom_counts, success, dim2, dim3

def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute 2D and 3D descriptors from NPZ.")
    # デフォルトを .npz に変更
    parser.add_argument("--input", default="data/processed/MoNA.npz")
    parser.add_argument("--output", default="data/processed/MoNA_rdkit-features.npz")
    parser.add_argument("--max-atoms", type=int, default=100)
    parser.add_argument("--max-embed-attempts", type=int, default=10) # 軽くするために回数を減らしています
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _disable_rdkit_logging()

    in_path = Path(args.input)
    out_path = Path(args.output)
    
    print(f"Reading {in_path} ...")
    try:
        # npzからsmilesキーを読み込む
        data = np.load(in_path, allow_pickle=True)
        if 'smiles' not in data:
             raise ValueError(f"Key 'smiles' not found in {in_path}. Keys: {list(data.keys())}")
        
        # numpy array -> list of strings
        smiles_list = data['smiles'].astype(str).tolist()
        print(f"Loaded {len(smiles_list)} molecules.")
        
    except Exception as e: 
        print(f"Error loading input: {e}")
        return

    # Process
    d2, d3, counts, success, dim2, dim3 = process_smiles(
        smiles_list, args.max_atoms, args.seed, args.max_embed_attempts
    )

    print(f"3D Success Rate: {success.mean()*100:.2f}%")
    
    print(f"Saving to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # NPZ保存
    np.savez_compressed(
        out_path,
        smiles=np.array(smiles_list, dtype=object),
        descriptors_2d=d2,
        descriptors_3d=d3,
        atom_counts=counts,
        success=success,
        dim_2d=np.int32(dim2),
        dim_3d=np.int32(dim3),
    )
    print("Done.")

if __name__ == "__main__":
    main()