# export_to_onnx.py
import torch
import torch.nn as nn
from collections import OrderedDict

# 假設您的模型定義檔是 hdr_transformer_ori.py
# 並且您已經根據之前的討論，知道要用 mlp_ratio=2
from models.hdr_transformerV10 import HDRTransformer 

def load_pytorch_model(model_path, device):
    """載入 PyTorch 模型並處理權重鍵名"""
    print("Loading PyTorch model...")
    # --- 1. 建立模型架構 (必須與訓練時完全相同) ---
    model = HDRTransformer(
                 img_size=128, patch_size=1, in_chans=3,
                 embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 ape=False, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv',
                 use_dynamic_tanh=True,
                 dyt_alpha_init_value=0.5
    ).to(device)
    model = nn.DataParallel(model)

    # --- 2. 載入權重 ---
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # 處理 'module.' 前綴問題
    is_parallel = all(k.startswith('module.') for k in state_dict.keys())
    if not is_parallel:
        print("Weight keys do not have 'module.' prefix. Adding it now.")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval() # 設定為評估模式
    print("PyTorch model loaded successfully.")
    return model.module # 回傳不含 DataParallel 的原始模型

def main():
    # --- 設定 ---
    pytorch_model_path = './checkpoints/V10-3/best_checkpoint.pth' # 您的 .pth 權重路徑
    onnx_model_path = './results/hdr_transformerV10_3.onnx' # 您要儲存的 .onnx 檔案路徑
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 載入 PyTorch 模型 ---
    model = load_pytorch_model(pytorch_model_path, device)

    # --- 準備虛擬輸入 (Dummy Input) ---
    # ONNX 匯出需要一個範例輸入來「追蹤」模型的運算圖
    # 您的模型需要三個輸入，且每個輸入是 6 通道
    batch_size = 1
    input_channels = 6
    height = 128
    width = 128
    
    # 建立三個符合模型輸入的虛擬張量
    dummy_input1 = torch.randn(batch_size, input_channels, height, width, device=device)
    dummy_input2 = torch.randn(batch_size, input_channels, height, width, device=device)
    dummy_input3 = torch.randn(batch_size, input_channels, height, width, device=device)
    dummy_inputs = (dummy_input1, dummy_input2, dummy_input3)

    print(f"Exporting model to {onnx_model_path}...")
    
    # --- 執行匯出 ---
    torch.onnx.export(
        model,                          # 要匯出的模型
        dummy_inputs,                   # 模型的虛擬輸入 (一個包含三個張量的元組)
        onnx_model_path,                # 儲存路徑
        export_params=True,             # 匯出時一併儲存權重
        opset_version=15,               # ONNX 的操作集版本，11或12是常用且穩定的選擇
        do_constant_folding=True,       # 是否執行常數折疊優化
        input_names=['input1', 'input2', 'input3'], # 為輸入節點命名 (非常重要)
        output_names=['output'],        # 為輸出節點命名 (非常重要)
        dynamic_axes={                  # 設定動態軸，讓模型可以接受不同大小的輸入
            'input1': {0: 'batch_size', 2: 'height', 3: 'width'},
            'input2': {0: 'batch_size', 2: 'height', 3: 'width'},
            'input3': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print("Model has been converted to ONNX successfully!")

if __name__ == '__main__':
    main()