#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
import time
import glob
import logging
import argparse

from torch.utils.data import DataLoader
#from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr #新版本
from tqdm import tqdm

from dataset.dataset_sig17V1 import SIG17_Test_Dataset
from visualize import visualize_and_save
from models.hdr_transformerV11 import HDRTransformer
print("Using NEW HDRTransformer from hdr_transformerV10.py")
# ====================================================

from train import test_single_img
from utils.utils import *

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset_dir", type=str, default='./data',
                        help='dataset directory')
parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
parser.add_argument('--num_workers', type=int, default=1, metavar='N',
                        help='number of workers to fetch data (default: 1)')
# DyT arguments
parser.add_argument('--use_dynamic_tanh', action='store_true', default=True,
                    help='model was trained with DynamicTanh instead of LayerNorm')
parser.add_argument('--dyt_alpha_init_value', type=float, default=0.5,
                    help='initial alpha value for DynamicTanh (mostly for model structure consistency)')
#visualization arguments
parser.add_argument('--visualize', action='store_true', default=False, help='Enable feature map visualization')
parser.add_argument('--vis_layers', nargs='+', default=[], help='A list of layer names to visualize, e.g., module.conv_first module.conv_after_body')

parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--pretrained_model', type=str, default='./checkpoints/V11_1/best_checkpoint.pth')
parser.add_argument('--test_best', action='store_true', default=False)
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default="./results/hdr_transformer")
parser.add_argument('--model_arch', type=int, default=0)

def main():
    # Settings
    args = parser.parse_args()

    # pretrained_model
    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.pretrained_model)

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # model architecture
    # 由於 V9 和 V10 的初始化參數相同，這裡不需要修改
    model_dict = {
        0: HDRTransformer(img_size=args.patch_size, embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6], mlp_ratio=2, in_chans=6, use_dynamic_tanh=args.use_dynamic_tanh,
            dyt_alpha_init_value=args.dyt_alpha_init_value),
    }
    model = model_dict[args.model_arch].to(device)
    model = nn.DataParallel(model)
    
    # 這裡的權重載入邏輯可以保持不變，因為它現在要載入的權重與 V10 模型架構是匹配的
    checkpoint = torch.load(args.pretrained_model)
    state_dict = checkpoint['state_dict']

    # 處理 "module." 前綴問題
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    # 假設儲存的權重不帶 'module.' 前綴
    # 如果您的權重檔本身就是由 DataParallel 保存的，可能需要調整此處邏輯
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            name = 'module.' + k
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    # 使用處理後的狀態字典加載模型
    model.load_state_dict(new_state_dict)
    print("成功加載權重")
    model.eval()

    datasets = SIG17_Test_Dataset(args.dataset_dir, args.patch_size)
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    visualize_once = False
    for idx, img_dataset in enumerate(datasets):
        if args.visualize and idx == 0 and args.vis_layers:
            print("========================================================================")
            print(f"將對場景 {idx} 的中心圖塊 (patch) 進行視覺化...")

            # 1. 從 img_dataset 中獲取所有 patches 和曝光時間
            all_patches = img_dataset.ldr_patches
            expo_times = img_dataset.expo_times

            # 2. 選擇一個代表性的 patch (例如，圖片最中間的那個 patch)
            center_patch_index = len(all_patches) // 2
            target_patch_group = all_patches[center_patch_index]  # 這是一個包含3張曝光的patch的列表

            # 3. 對這個選中的 patch 進行與模型訓練時完全相同的預處理
            input_tensors_for_vis = []
            for i in range(len(target_patch_group)):
                patch_np = target_patch_group[i]
                linear_img = ldr_to_hdr(patch_np, expo_times[i], 2.2)
                pre_img = np.concatenate((linear_img, patch_np), axis=2)
                tensor = torch.from_numpy(pre_img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
                input_tensors_for_vis.append(tensor)
            
            # 4. 準備用於疊加的背景圖 (就是中間曝光的那個 patch)
            background_np_patch = target_patch_group[1]
            
            # 檔案儲存路徑
            vis_save_dir = os.path.join(args.save_dir, 'visualizations')
            os.makedirs(vis_save_dir, exist_ok=True)

            # 5. 迴圈處理所有要視覺化的層
            for layer_name in args.vis_layers:
                save_path = os.path.join(vis_save_dir, f'scene_{idx:03d}_patch_{center_patch_index}_{layer_name.replace(".", "_")}.jpg')
                visualize_and_save(
                    model=model,
                    input_tensors=tuple(input_tensors_for_vis),
                    background_image_numpy=background_np_patch, # 使用 patch 作為背景
                    target_layer_name=layer_name,
                    save_path=save_path
                )

            print(f"場景 {idx} 的視覺化任務完成。")
            print("========================================================================")
    # ==========================================================================
        pred_img, label = test_single_img(model, img_dataset, device)
        pred_hdr = pred_img.copy()
        pred_hdr = pred_hdr.transpose(1, 2, 0)[..., ::-1]
        # psnr-l and psnr-\mu
        scene_psnr_l = compare_psnr(label, pred_img, data_range=1.0)
        label_mu = range_compressor(label)
        pred_img_mu = range_compressor(pred_img)
        scene_psnr_mu = compare_psnr(label_mu, pred_img_mu, data_range=1.0)
        # ssim-l
        pred_img = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
        label = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
        scene_ssim_l = calculate_ssim(pred_img, label)
        # ssim-\mu
        pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu)

        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)

        # save results
        if args.save_results:
            if not osp.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_hdr(os.path.join(args.save_dir, '{}_pred.hdr'.format(idx)), pred_hdr)

    print("Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(psnr_mu.avg, psnr_l.avg))
    print("Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssim_mu.avg, ssim_l.avg))
    print(">>>>>>>>> Finish Testing >>>>>>>>>")


if __name__ == '__main__':
    main()