# demo_onnx_full.py
import os
import argparse
import logging
import time

import onnxruntime as ort
import numpy as np
import cv2
from tqdm import tqdm
from dataset.dataset_sig17V1 import SIG17_Test_Dataset


# --- 輔助函式 ---

def setup_logging(save_dir):
    """設定日誌記錄器"""
    log_path = os.path.join(save_dir, "demo_onnx_info.log")
    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        filemode='w'
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    return logger

def save_img_png(filepath, img_array):
    """將 float32 範圍 [0, 1] 的 numpy 陣列儲存為 PNG"""
    img_to_save = np.clip(img_array, 0, 1)
    img_to_save = (img_to_save * 255.0).astype(np.uint8)
    img_to_save = np.transpose(img_to_save, (1, 2, 0))
    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR) # <-- 已修正為正確的旗標
    cv2.imwrite(filepath, img_to_save)

def load_onnx_session(onnx_path):
    """載入 ONNX 模型並建立推論會話"""
    logger = logging.getLogger()
    logger.info(f"Loading ONNX model from {onnx_path}...")
    try:
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        logger.info(f"ONNX model loaded successfully. Using provider: {session.get_providers()[0]}")
        return session
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        raise


def run_onnx_demo(session, args):
    """
    執行完整的 ONNX Demo 流程，包含處理場景和拼接圖塊。
    """
    logger = logging.getLogger()
    logger.info('Starting ONNX Demo with correct two-loop structure...')

    # 外層迴圈：遍歷所有場景
    # `scene_dataset` 現在是 Img_Dataset 物件
    scene_generator = SIG17_Test_Dataset(args.dataset_dir, args.patch_size)
    
    # 由於生成器沒有 len()，我們需要手動計算或 просто迭代
    # 為了進度條，我們先將其轉換為列表
    all_scenes = list(scene_generator)
    pbar = tqdm(enumerate(all_scenes), total=len(all_scenes), desc="Processing Scenes")

    total_inference_time = 0
    total_patches = 0

    for scene_idx, scene_dataset in pbar:
        pbar.set_description(f"Scene {scene_idx+1}/{len(all_scenes)}")
        
        # 內層迴圈：遍歷單一場景中的所有圖塊
        for patch_idx in range(len(scene_dataset)):
            
            # 1. 從 scene_dataset 物件中獲取單一圖塊的資料
            # 這裡的 patch_sample 是一個字典 {'input0': ..., 'input1': ..., 'input2': ...}
            patch_sample = scene_dataset[patch_idx]

            # 2. 從字典中提取輸入張量
            l_img_tensor = patch_sample['input0']
            m_img_tensor = patch_sample['input1']
            h_img_tensor = patch_sample['input2']

            #l_img_tensor_3ch = l_img_tensor[:3, :, :]
            #m_img_tensor_3ch = m_img_tensor[:3, :, :]
            #h_img_tensor_3ch = h_img_tensor[:3, :, :]

            # 3. 將 PyTorch Tensors 轉換為 NumPy Arrays
            np_input1 = l_img_tensor.cpu().numpy()
            np_input2 = m_img_tensor.cpu().numpy()
            np_input3 = h_img_tensor.cpu().numpy()

            # 確保 batch 維度存在
            if np_input1.ndim == 3: np_input1 = np.expand_dims(np_input1, 0)
            if np_input2.ndim == 3: np_input2 = np.expand_dims(np_input2, 0)
            if np_input3.ndim == 3: np_input3 = np.expand_dims(np_input3, 0)

            # --- 原始錯誤的程式碼 ---
            # 4. 準備輸入字典並執行推論
            onnx_inputs = {
                'input1': np_input1,
                'input2': np_input2,
                'input3': np_input3
            }

            start_time = time.time()
            onnx_outputs = session.run(['output'], onnx_inputs)
            inference_time = time.time() - start_time

            total_inference_time += inference_time
            total_patches += 1
            
            # 5. 將單一圖塊的推論結果存回 scene_dataset 物件中
            # 移除 batch 維度，並確保是 float32
            patch_result_np = onnx_outputs[0].astype(np.float32)
            scene_dataset.update_result(patch_result_np)
        
        # --- 內層迴圈結束：一個場景的所有圖塊都已處理完畢 ---

        # 6. 重建(拼接)單一場景的完整預測影像
        if args.save_results:
            logger.info(f"Rebuilding result for scene {scene_idx+1}...")
            rebuilt_image, _ = scene_dataset.rebuild_result()
            
            # 使用外層迴圈的索引值 scene_idx 來命名，更簡單且不會出錯
            # f"{scene_idx+1:03d}" 會產生 "001", "002", ... 這樣的字串
            scene_name = f"{scene_idx+1:03d}"
            output_filename = os.path.join(args.save_dir, f'{scene_name}_pred.png')
            
            save_img_png(output_filename, rebuilt_image)

    avg_latency = (total_inference_time / total_patches) * 1000 if total_patches > 0 else 0
    logger.info(f"Demo finished. Processed {len(all_scenes)} scenes and {total_patches} patches in total.")
    logger.info(f"Average ONNX Inference Latency: {avg_latency:.2f} ms/patch")


def main():
    # --- 參數設定 ---
    parser = argparse.ArgumentParser(description="Full ONNX Demo for Ka")
    parser.add_argument("--onnx_model", type=str, default='./results/hdr_transformerV11.onnx', help='Path to the ONNX model file.')
    parser.add_argument("--dataset_dir", type=str, default='./data', help='Dataset directory')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for model input (should match training)')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save output images')
    parser.add_argument('--save_dir', type=str, default="./results/onnx_demo_full", help='Directory to save results')
    args = parser.parse_args()

    # --- 初始化 ---
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logging(args.save_dir)
    logger.info(">>>>>>>>> Start Full ONNX Demo >>>>>>>>>")
    logger.info(f"Arguments: {args}")

    # --- 載入 ONNX 模型 ---
    session = load_onnx_session(args.onnx_model)
    
    # --- 執行 Demo ---
    run_onnx_demo(session, args)

    logger.info(">>>>>>>>> Demo Finished >>>>>>>>>")

if __name__ == '__main__':
    main()