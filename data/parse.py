import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

# --- 設定パラメータ ---
INPUT_SDF = "data/raw/MoNA.sdf"       # 入力ファイルパス
OUTPUT_PARQUET = "data/processed/MoNA.parquet" # 出力ファイルパス

MAX_MZ = 2000                      # m/z範囲 (0〜2000)
SANITIZE = True                    # Strictモード (RDKitで構造エラーなら捨てる)
FILTER_EI = True                   # EI関連のキーワードが含まれるかチェックするか

# --- 官能基定義 (SMARTS) ---
SMARTS_PATTERNS = {
    'has_ether':      '[OD2]([#6])[#6]',
    'has_carbonyl':   '[CX3]=[OX1]',
    'has_alcohol':    '[#6][OX2H]',
    'has_amine':      '[NX3;!$(NC=O)]',
    'has_cyano':      '[NX1]#[CX2]',
    'has_amide':      '[NX3][CX3](=[OX1])',
    'has_carboxylic': '[CX3](=O)[OX2H1]',
}
MOL_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS_PATTERNS.items()}

def parse_spectrum(spec_str, max_mz=2000):
    """
    スペクトル文字列をパースし、(正規化配列, ピーク数, 総強度) を返す
    """
    if not spec_str:
        return None, 0, 0.0

    spectrum = np.zeros(max_mz + 1, dtype=np.float32)
    lines = spec_str.strip().split('\n')
    
    peak_count = 0
    total_intensity = 0.0
    max_intensity = 0.0

    try:
        for line in lines:
            line = line.strip()
            if not line: continue
            
            parts = line.split()
            if len(parts) < 2: continue
            
            mz = float(parts[0])
            intensity = float(parts[1])
            
            if mz > max_mz or mz < 0:
                continue
            
            idx = int(round(mz))
            
            # Max Pooling & Stats accumulation
            if intensity > spectrum[idx]:
                spectrum[idx] = intensity
            
            # 統計用には生データを加算
            total_intensity += intensity
            peak_count += 1
            
            if intensity > max_intensity:
                max_intensity = intensity
        
        # 正規化
        if max_intensity > 0:
            spectrum = spectrum / max_intensity
            
        return spectrum, peak_count, total_intensity

    except:
        return None, 0, 0.0

def get_molecular_flags(mol):
    """フラグ抽出"""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    bonds = [b.GetBondType() for b in mol.GetBonds()]
    total_hydrogens = sum(a.GetTotalNumHs() for a in mol.GetAtoms())

    flags = {}
    flags['num_O'] = atoms.count('O')
    flags['num_N'] = atoms.count('N')
    flags['num_F'] = atoms.count('F')
    flags['has_C'] = 1 if 'C' in atoms else 0
    flags['has_O'] = 1 if 'O' in atoms else 0
    flags['has_N'] = 1 if 'N' in atoms else 0
    flags['has_F'] = 1 if 'F' in atoms else 0
    flags['has_H'] = 1 if total_hydrogens > 0 else 0
    
    flags['has_single']   = 1 if Chem.BondType.SINGLE in bonds else 0
    flags['has_double']   = 1 if Chem.BondType.DOUBLE in bonds else 0
    flags['has_triple']   = 1 if Chem.BondType.TRIPLE in bonds else 0
    flags['has_aromatic'] = 1 if Chem.BondType.AROMATIC in bonds else 0
    flags['has_ring'] = 1 if mol.GetRingInfo().NumRings() > 0 else 0

    for name, pattern in MOL_PATTERNS.items():
        flags[name] = 1 if mol.HasSubstructMatch(pattern) else 0

    return flags

def is_ei_data(mol):
    """
    メタデータからEI(電子イオン化)っぽいか判定する簡易フィルタ
    タグがない場合はTrue(除外しない)とみなす
    """
    # チェックするタグのリスト
    tags_to_check = ["INSTRUMENT TYPE", "IONIZATION", "FRAGMENTATION MODE", "ION MODE"]
    
    found_tag = False
    is_ei = False
    
    for tag in tags_to_check:
        if mol.HasProp(tag):
            val = mol.GetProp(tag).upper()
            found_tag = True
            # EI, ELECTRON, 70EV などのキーワードが含まれるか
            # もしくは Ion Mode が P (Positive) であるか (GC-MSのEIは通常Positive)
            if any(x in val for x in ["EI", "ELECTRON", "70", "P", "POS"]):
                is_ei = True
    
    # タグが全くない、あるいは判定できない場合は、ユーザを信じてTrueを返す
    if not found_tag:
        return True
        
    return is_ei

def process_sdf_best_selection(sdf_path, output_path):
    # 重複排除用辞書: { inchikey: {data: row_dict, score: (peaks, intensity)} }
    best_records = {}
    
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=SANITIZE)
    print(f"Reading {sdf_path} with STRICT mode...")
    
    processed_count = 0
    skipped_struct = 0
    skipped_spec = 0
    error_count = 0
    last_error = None
    
    for mol in tqdm(suppl):
        if mol is None:
            skipped_struct += 1
            continue
            
        try:
            # 1. 構造情報の取得
            inchikey = Chem.MolToInchiKey(mol)
            smiles = Chem.MolToSmiles(mol)
            if not inchikey or not smiles:
                skipped_struct += 1
                continue

            # 2. EIフィルタ (オプション)
            if FILTER_EI and not is_ei_data(mol):
                continue

            # 3. スペクトル処理
            if not mol.HasProp("MASS SPECTRAL PEAKS"):
                skipped_spec += 1
                continue
                
            raw_spec = mol.GetProp("MASS SPECTRAL PEAKS")
            spectrum_array, peak_count, total_intensity = parse_spectrum(raw_spec, MAX_MZ)
            
            if spectrum_array is None or peak_count == 0:
                skipped_spec += 1
                continue

            # 4. スコアリング (ピーク数優先、次に総強度)
            current_score = (peak_count, total_intensity)
            
            # 5. ベスト更新ロジック
            # 既に同じInChIKeyがあるか？
            if inchikey in best_records:
                existing_score = best_records[inchikey]['score']
                # 現在の方がスコアが高ければ更新 (タプルの比較は左から順に行われる)
                if current_score > existing_score:
                    # フラグ取得 (更新時のみ計算してコスト削減…と言いたいが構造変わらないのでどちらでも可)
                    # ここでは新しいmolからフラグを取り直す
                    flags = get_molecular_flags(mol)
                    row = {
                        'smiles': smiles,
                        'inchikey': inchikey,
                        'spectrum': spectrum_array,
                        **flags
                    }
                    best_records[inchikey] = {'data': row, 'score': current_score}
            else:
                # 新規登録
                flags = get_molecular_flags(mol)
                row = {
                    'smiles': smiles,
                    'inchikey': inchikey,
                    'spectrum': spectrum_array,
                    **flags
                }
                best_records[inchikey] = {'data': row, 'score': current_score}
                
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            last_error = str(e)
            continue

    print("-" * 30)
    print(f"Total processed: {processed_count}")
    print(f"Skipped (Struct Error): {skipped_struct}")
    print(f"Skipped (No Spec): {skipped_spec}")
    if error_count:
        print(f"Skipped (Exception): {error_count} | Last error: {last_error}")
    print(f"Unique Molecules (InChIKey): {len(best_records)}")
    
    # DataFrame化
    if len(best_records) > 0:
        # 辞書からデータ部分のみを取り出す
        final_data = [item['data'] for item in best_records.values()]
        df = pd.DataFrame(final_data)

        # 通し番号付与
        df.insert(0, "entry_id", np.arange(len(df), dtype=np.int64))

        # ndarray列をそのままParquetに書くと失敗するのでリスト化
        df['spectrum'] = df['spectrum'].apply(lambda x: x.tolist())
        
        # Parquet保存
        df.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"✅ Saved to {output_path}. Shape: {df.shape}")

        # 簡易チェック
        sample_spec = df.iloc[0]['spectrum']
        if isinstance(sample_spec, list):
            sample_spec = np.array(sample_spec, dtype=np.float32)
        print("Sample Spectrum Max:", sample_spec.max() if len(sample_spec) > 0 else None)
        print("Sample Spectrum Len:", len(sample_spec))
    else:
        print("❌ No valid data found.")

if __name__ == "__main__":
    process_sdf_best_selection(INPUT_SDF, OUTPUT_PARQUET)
