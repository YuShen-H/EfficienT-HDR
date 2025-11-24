# run_visualizations.py
import subprocess
import sys
import os

# --- 使用者設定 ---
# sys.executable 會自動使用當前執行此腳本的 python.exe，確保環境正確無誤！
PYTHON_EXE = sys.executable
SCRIPT_PATH = "testV10.py"
SAVE_DIR = os.path.join("results", "hdr_transformer")

# --- 推薦視覺化的層列表 ---
layers_to_visualize = [
    "module.conv_first",
    "module.layers.0.dilated_conv",
    "module.layers.2.dilated_conv",
    "module.layers.0.residual_group.blocks.0.lce",
    "module.conv_after_body"
]# run_visualizations.py (高效版)
import subprocess
import sys
import os

# --- 使用者設定 ---
# sys.executable 會自動使用當前執行此腳本的 python.exe，確保環境正確無誤！
PYTHON_EXE = sys.executable
SCRIPT_PATH = "testV10.py"
MODEL_PATH = os.path.join("checkpoints", "V10-1", "best_checkpoint.pth")
SAVE_DIR = os.path.join("results", "hdr_transformer")

# --- 推薦視覺化的層列表 ---
layers_to_visualize = [
    "module.conv_first",
    "module.layers.0.dilated_conv",
    "module.layers.2.dilated_conv",
    "module.layers.0.residual_group.blocks.0.lce",
    "module.conv_after_body"
]

print("準備開始自動化特徵圖視覺化 (高效 Python 腳本版)...")
print(f"將會執行一次完整的測試，並在過程中產生 {len(layers_to_visualize)} 張視覺化圖。")
print("=" * 70)


# --- 執行指令 (只執行一次) ---

# 建構基礎指令
command = [
    PYTHON_EXE,
    SCRIPT_PATH,
    "--pretrained_model", MODEL_PATH,
    "--save_dir", SAVE_DIR,
    "--save_results",
    "--visualize",
    "--vis_layers"  # 使用新的複數參數
]

# 將層列表中的所有層名都加到指令的末尾
command.extend(layers_to_visualize)

# 執行完整的指令
# 例如，它實際執行的會像這樣：
# python testV10.py --pretrained_model ... --vis_layers module.conv_first module.layers.0.dilated_conv ...
subprocess.run(command)


print("=" * 70)
print("所有測試與視覺化任務已全部執行完畢！")

print("準備開始自動化特徵圖視覺化 (Python 腳本版)...")

# --- 自動執行迴圈 ---
for layer in layers_to_visualize:
    print("=" * 70)
    print(f"正在處理層: {layer}")

    command = [
        PYTHON_EXE,
        SCRIPT_PATH,
        "--pretrained_model", MODEL_PATH,
        "--save_dir", SAVE_DIR,
        "--save_results",
        "--visualize",
        "--vis_layer", layer
    ]

    # 執行指令
    subprocess.run(command)

print("=" * 70)
print("所有推薦層的視覺化已全部執行完畢！")