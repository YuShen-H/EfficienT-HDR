#-*- coding:utf-8 -*-
import os
import os.path as osp
import argparse
import logging
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import cv2

# 假設這些模組與您的 testV9.py 位於相同環境中
from models.hdr_transformerV10 import HDRTransformer
from dataset.dataset_sig17V1 import SIG17_Test_Dataset
from utils.utils import save_hdr # 從您的 utils 導入
from train import test_single_img # 從您的 train.py 導入

# 使用 fvcore 計算 FLOPs，若未安裝請執行: pip install fvcore
from fvcore.nn import FlopCountAnalysis

# --- Main Functions ---

def setup_logging(save_dir):
    """設定日誌記錄器"""
    log_path = os.path.join(save_dir, "demo_info.log")
    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        filemode='w'
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler()) # 同時輸出到控制台
    return logger

# <--- 新增函式：只建立模型架構，不載入權重 --->
def create_hdr_transformer_model(args, device):
    """僅建立 HDRTransformer 模型架構，用於 FLOPs 計算。"""
    logger = logging.getLogger()
    logger.info(f"Creating model architecture (model_arch: {args.model_arch}) without loading weights.")
    
    # --- 模型架構 (與原函式相同) ---
    model_dict = {
        0: HDRTransformer(
                 img_size=128, patch_size=1, in_chans=3,
                 embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 ape=False, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv',
                 use_dynamic_tanh=True,
                 dyt_alpha_init_value=0.5,
        ),
    }
    model = model_dict[args.model_arch].to(device)
    model = nn.DataParallel(model)
    model.eval() # 設定為評估模式
    return model

def load_hdr_transformer_model(args, device):
    """載入預訓練的 HDRTransformer 模型"""
    logger = logging.getLogger()
    
    # <--- 修改：呼叫新建的 create_hdr_transformer_model 來建立模型 --->
    model = create_hdr_transformer_model(args, device)
    
    logger.info(f"Loading weights from: {args.pretrained_model}")
    if not os.path.exists(args.pretrained_model):
        logger.error(f"Checkpoint file not found at: {args.pretrained_model}")
        raise FileNotFoundError(f"Checkpoint file not found at: {args.pretrained_model}")

    # --- 載入權重 ---
    checkpoint = torch.load(args.pretrained_model, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint) # 更穩健地獲取 state_dict
    
    # 檢查鍵是否以 'module.' 開頭
    is_parallel = all(k.startswith('module.') for k in state_dict.keys())
    
    if not is_parallel and isinstance(model, nn.DataParallel):
        logger.info("Weight keys do not have 'module.' prefix. Adding it now.")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    logger.info("Successfully loaded pretrained weights.")
    model.eval()
    return model

def save_img_png(filepath, img):
    """將 float32 範圍 [0, 1] 的圖像儲存為 PNG"""
    img_to_save = np.transpose(img, (1, 2, 0))
    img_to_save = np.clip(img_to_save * 255.0, 0, 255).astype(np.uint8)
    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_to_save)

def run_demo(model, device, args):
    """執行 Demo 流程"""
    logger = logging.getLogger()
    logger.info('Starting Demo...')
    datasets = SIG17_Test_Dataset(args.dataset_dir, args.patch_size)
    pbar = tqdm(enumerate(datasets), desc="Processing images", total=len(datasets))

    for idx, img_dataset in pbar:
        pred_img, _ = test_single_img(model, img_dataset, device)
        pred_hdr = pred_img.copy()
        pred_hdr = pred_hdr.transpose(1, 2, 0)[..., ::-1]

        if args.save_results:
            hdr_filename = os.path.join(args.save_dir, f'{idx:03d}_pred.hdr')
            save_hdr(hdr_filename, pred_hdr)
            png_filename = os.path.join(args.save_dir, f'{idx:03d}_pred.png')
            save_img_png(png_filename, pred_img)
            
    logger.info(f"Demo finished. Results saved in {args.save_dir}")

def measure_network_efficiency(model, device, args):
    """測量模型的參數、計算量和速度"""
    logger = logging.getLogger()
    logger.info("\n--- Network Efficiency Analysis ---")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total trainable parameters: {total_params / 1e6:.2f} M')

    dummy_input_res = (6, args.patch_size, args.patch_size)
    dummy_input1 = torch.randn(1, *dummy_input_res).to(device)
    dummy_input2 = torch.randn(1, *dummy_input_res).to(device)
    dummy_input3 = torch.randn(1, *dummy_input_res).to(device)
    
    try:
        # <--- 修改：確保在 DataParallel 模型上正確使用 flops --->
        flops_model = model.module if isinstance(model, nn.DataParallel) else model
        flops = FlopCountAnalysis(flops_model, (dummy_input1, dummy_input2, dummy_input3))
        total_flops = flops.total()
        logger.info(f'FLOPs (for {args.patch_size}x{args.patch_size} input): {total_flops / 1e9:.2f} G')
    except Exception as e:
        logger.warning(f"Could not calculate FLOPs: {e}")

    iterations = 100
    total_time = 0
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input1, dummy_input2, dummy_input3)
        
        start_time = time.time()
        for _ in range(iterations):
            _ = model(dummy_input1, dummy_input2, dummy_input3)
        end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    logger.info(f"Average inference time: {avg_time_ms:.2f} ms/image ({iterations} iterations)")
    logger.info("--- End of Analysis ---\n")

def main():
    parser = argparse.ArgumentParser(description="HDR Transformer Demo and Evaluation")
    parser.add_argument("--dataset_dir", type=str, default='./data', help='dataset directory')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--patch_size', type=int, default=128, help='patch size for model input')
    parser.add_argument('--pretrained_model', type=str, default='./checkpoints/V9-2/best_checkpoint.pth', help='path to pretrained model')
    parser.add_argument('--save_results', action='store_true', help='save output images')
    parser.add_argument('--save_dir', type=str, default="./results/hdr_transformer_demo", help='directory to save results')
    parser.add_argument('--model_arch', type=int, default=0, help='model architecture index')
    parser.add_argument('--use_dynamic_tanh', action='store_true', default=True)
    parser.add_argument('--dyt_alpha_init_value', type=float, default=0.5)
    # <--- 新增參數：用於控制是否只測試FLOPs --->
    parser.add_argument(
        '--flops', 
        action='store_true', 
        help='If specified, only initialize the model to measure FLOPs without loading a checkpoint.'
    )
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logging(args.save_dir)
    logger.info(f"Arguments: {args}")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    logger.info(f"Using device: {device}")

    # <--- 修改主流程：根據 measure_flops_only 決定執行路徑 --->
    if args.flops:
        logger.info(">>>>>>>>> Measuring FLOPs only (no checkpoint loading) >>>>>>>>>")
        # 1. 只建立模型，不載入權重
        model = create_hdr_transformer_model(args, device)
        # 2. 直接評估效能
        measure_network_efficiency(model, device, args)
    else:
        logger.info(">>>>>>>>> Start HDR Transformer Demo >>>>>>>>>")
        # --- 正常流程 ---
        # 1. 載入模型和權重
        model = load_hdr_transformer_model(args, device)
        # 2. 執行 Demo 推論
        run_demo(model, device, args)
        # 3. 評估模型效能
        measure_network_efficiency(model, device, args)

    logger.info(">>>>>>>>> Process Finished >>>>>>>>>")

if __name__ == '__main__':
    main()