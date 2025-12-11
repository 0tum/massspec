import sys
import subprocess
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- 設定 ---
INPUT_NPZ = "data/processed/MoNA.npz"
OUTPUT_NPZ = "data/processed/MoNA_feature.npz"
WORKER_SCRIPT = "worker.py"
TIMEOUT_SEC = 10  # 1分子あたりの制限時間
DIM_2D = 208      # RDKitのバージョンによりますが一旦固定か、最初の1個で判定
DIM_3D = 0        # 最初の成功時に確定させます

# 【修正1】worker.py のパスを、このスクリプト(main_runner.py)と同じ場所から探すように変更
CURRENT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = str(CURRENT_DIR / "worker.py")

def get_dims_from_dummy():
    """ダミー分子で次元数を確認"""
    print(f"DEBUG: Using worker script at: {WORKER_SCRIPT}")
    
    if not Path(WORKER_SCRIPT).exists():
        raise FileNotFoundError(f"Worker script not found at {WORKER_SCRIPT}")

    cmd = [sys.executable, WORKER_SCRIPT, "CC", "42"]
    
    # 【修正2】エラー時に詳細を表示する
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    if res.returncode != 0:
        print("❌ Dummy run failed!")
        print("--- STDERR ---")
        print(res.stderr)
        print("--------------")
        raise RuntimeError("Worker script failed to run.")

    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        print("❌ JSON Decode Error!")
        print(f"STDOUT content: '{res.stdout}'")
        print(f"STDERR content: '{res.stderr}'")
        raise

    d2 = len(data['d2'])
    d3 = len(data['d3'])
    return d2, d3

def main():
    print("Checking dimensions...")
    try:
        dim2, dim3 = get_dims_from_dummy()
        print(f"Detected Dimensions -> 2D: {dim2}, 3D: {dim3}")
    except Exception as e:
        print(f"Error checking dimensions: {e}")
        return

    # データの読み込み
    print(f"Loading {INPUT_NPZ}...")
    data = np.load(INPUT_NPZ, allow_pickle=True)
    smiles_list = data['smiles'].astype(str)
    n = len(smiles_list)
    
    # 結果格納用配列
    arr_2d = np.zeros((n, dim2), dtype=np.float32)
    arr_3d = np.zeros((n, dim3), dtype=np.float32)
    atom_counts = np.zeros(n, dtype=np.int32)
    success = np.zeros(n, dtype=bool)
    
    # 進捗バー
    pbar = tqdm(enumerate(smiles_list), total=n, unit="mol")
    
    # --- メインループ ---
    for i, smi in pbar:
        # worker.py をサブプロセスとして実行
        cmd = [sys.executable, WORKER_SCRIPT, smi, "42"]
        
        try:
            # ここが重要: timeoutを設定して外部プロセスを呼ぶ
            # capture_output=Trueで標準出力を受け取る
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)
            
            # 正常終了した場合、出力をパース
            output_json = proc.stdout.strip()
            if not output_json:
                raise ValueError("Empty output")
                
            res = json.loads(output_json)
            
            if res['status'] == 'success':
                # データ格納
                d2 = res['d2']
                d3 = res['d3']
                
                # 次元チェック (念のため)
                if len(d3) != dim3:
                    # パディングなどの処理が必要ならここで行う
                    # 今回はworkerが可変で返してくる設定なので、親でslice/padする
                    curr_d3 = np.array(d3, dtype=np.float32)
                    if len(curr_d3) > dim3:
                        arr_3d[i] = curr_d3[:dim3]
                    else:
                        arr_3d[i, :len(curr_d3)] = curr_d3
                else:
                    arr_3d[i] = d3
                    
                arr_2d[i] = d2
                atom_counts[i] = res['atoms']
                success[i] = True
            
            elif res['status'] == 'fail_3d':
                # 2Dだけ取れた場合
                arr_2d[i] = res['d2']
                success[i] = False
                pbar.set_postfix({"last": "3D_Fail"})
            else:
                # その他のエラー
                success[i] = False
                # pbar.set_postfix({"last": res.get('msg', 'Error')})

        except subprocess.TimeoutExpired:
            # タイムアウト発生時: プロセスはOSによってキルされる
            success[i] = False
            pbar.set_postfix({"last": "TIMEOUT"})
            # 続きから処理可能
            
        except Exception as e:
            # JSONパースエラーなど
            success[i] = False
            pbar.set_postfix({"last": "Crash"})

        # 定期保存 (オプション: 1000件ごとに保存するなど)
        if (i + 1) % 1000 == 0:
            pass # ここで中間保存しても良い

    # --- 保存 ---
    print(f"\nSuccess rate: {success.mean()*100:.1f}%")
    print(f"Saving to {OUTPUT_NPZ}...")
    np.savez_compressed(
        OUTPUT_NPZ,
        smiles=smiles_list,
        descriptors_2d=arr_2d,
        descriptors_3d=arr_3d,
        atom_counts=atom_counts,
        success=success,
        dim_2d=dim2,
        dim_3d=dim3
    )
    print("Done!")

if __name__ == "__main__":
    main()