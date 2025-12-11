import sys
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdMolDescriptors
from rdkit import RDLogger

# ログ抑制
RDLogger.DisableLog("rdApp.*")

def calc_2d(mol):
    vals = []
    for _, func in Descriptors.descList:
        try:
            v = func(mol)
            vals.append(float(v) if np.isfinite(v) else 0.0)
        except:
            vals.append(0.0)
    return vals

def calc_3d(mol, dim3=None):
    # --- ここで各種3D記述子を計算 ---
    # ※長くなるので前回と同じロジックを簡略化して書きます
    feats = []
    feats.append(Descriptors3D.PMI1(mol))
    feats.append(Descriptors3D.PMI2(mol))
    feats.append(Descriptors3D.PMI3(mol))
    feats.append(Descriptors3D.NPR1(mol))
    feats.append(Descriptors3D.NPR2(mol))
    feats.append(Descriptors3D.RadiusOfGyration(mol))
    feats.append(Descriptors3D.InertialShapeFactor(mol))
    feats.append(Descriptors3D.SpherocityIndex(mol))
    feats.extend(rdMolDescriptors.CalcMORSE(mol))
    feats.extend(rdMolDescriptors.CalcRDF(mol))
    feats.extend(rdMolDescriptors.CalcWHIM(mol))
    feats.extend(rdMolDescriptors.CalcGETAWAY(mol))
    feats.extend(rdMolDescriptors.CalcAUTOCORR3D(mol))
    
    # 欠損値埋め
    feats = [float(f) if np.isfinite(f) else 0.0 for f in feats]
    
    # 次元合わせ
    if dim3 and len(feats) < dim3:
        feats += [0.0] * (dim3 - len(feats))
    elif dim3 and len(feats) > dim3:
        feats = feats[:dim3]
        
    return feats

def run(smiles, max_atoms, seed, max_attempts, dim3_target):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"status": "error", "msg": "Invalid SMILES"}

        # 2D計算
        desc2d = calc_2d(mol)

        # 原子数チェック
        mol_h = Chem.AddHs(mol)
        if mol_h.GetNumAtoms() > max_atoms:
            return {"status": "error", "msg": "Too large"}

        # 3D埋め込み
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1 # 必ずシングルスレッド
        
        res = -1
        for i in range(max_attempts):
            params.randomSeed = seed + i
            res = AllChem.EmbedMolecule(mol_h, params)
            if res == 0: break
            mol_h.RemoveAllConformers()
        
        if res != 0:
            return {"status": "fail_3d", "d2": desc2d}

        try:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        except:
            pass
            
        desc3d = calc_3d(mol_h, dim3_target)
        
        return {
            "status": "success",
            "d2": desc2d,
            "d3": desc3d,
            "atoms": mol_h.GetNumAtoms()
        }

    except Exception as e:
        return {"status": "error", "msg": str(e)}

if __name__ == "__main__":
    # コマンドライン引数: SMILES, Seed
    # 例: python worker.py "CCC" 42
    try:
        smiles = sys.argv[1]
        seed = int(sys.argv[2])
        # 次元数は固定値として持つか、別途渡す（ここでは簡易的に固定長想定のロジックを入れるか、
        # もしくはパディングなしで返して親で処理するか。今回は簡易化のため親でパディングさせます）
        
        result = run(smiles, max_atoms=100, seed=seed, max_attempts=10, dim3_target=None)
        
        # JSONとして標準出力に吐く
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"status": "error", "msg": str(e)}))