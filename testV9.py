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
from models.hdr_transformerV9 import HDRTransformer
print("Using NEW HDRTransformer from hdr_transformer.py")
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
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--pretrained_model', type=str, default='./checkpoints/best_checkpoint.pth')
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
    model_dict = {
        0: HDRTransformer(img_size=args.patch_size, embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6], mlp_ratio=2, in_chans=6, use_dynamic_tanh=args.use_dynamic_tanh,
            dyt_alpha_init_value=args.dyt_alpha_init_value),
    }
    model = model_dict[args.model_arch].to(device)
    model = nn.DataParallel(model)
    #model.load_sjotate_dict(torch.load(args.pretrained_model)['state_dict'])
    checkpoint = torch.load(args.pretrained_model)
    state_dict = checkpoint['state_dict']

    # 處理 "module." 前綴問題
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
    # 如果參數鍵不以 "module." 開頭，但模型使用了 DataParallel
        name = 'module.' + k  # 添加 "module." 前綴
        new_state_dict[name] = v

# 使用處理後的狀態字典加載模型
    model.load_state_dict(new_state_dict)
    print("成功加載權重")
    model.eval()

    datasets = SIG17_Test_Dataset(args.dataset_dir, args.patch_size)
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    for idx, img_dataset in enumerate(datasets):
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




