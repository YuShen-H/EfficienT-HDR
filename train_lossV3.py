# -*- coding:utf-8 -*-
import os
import time
import argparse
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataset.dataset_sig17V1 import SIG17_Training_Dataset, SIG17_Validation_Dataset, SIG17_Test_Dataset
from models.loss import L1MuLoss, JointReconPerceptualLoss
from models.hdr_transformerV10 import HDRTransformer
from utils.utils import *



def get_args():
    parser = argparse.ArgumentParser(description='HDR-Transformer',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_dir", type=str, default='./data',
                        help='dataset directory'),
    parser.add_argument('--patch_size', type=int, default=128),
    parser.add_argument("--sub_set", type=str, default='sig17_training_crop128_stride64',
                        help='dataset directory')
    parser.add_argument('--logdir', type=str, default='./checkpoints',
                        help='target log directory')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    ############################################ DyT arguments################################################################
    parser.add_argument('--use_dynamic_tanh', action='store_true', default=True,
                        help='use DynamicTanh instead of LayerNorm')
    parser.add_argument('--dyt_alpha_init_value', type=float, default=0.5,
                        help='initial alpha value for DynamicTanh')
    # Training
    parser.add_argument('--resume', type=str, default=None,
                        help='load model from a .pth file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--init_weights', action='store_true', default=False,
                        help='init model weights')
    parser.add_argument('--loss_func', type=int, default=1,
                        help='loss functions for training')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=2e-2, # 新增 weight_decay 參數
                        help='AdamW weight decay (default: 0.02)')
    #cosine annealing
    #parser.add_argument('--t_max', type=int, default=85,
    #                    help='T_max for CosineAnnealingLR (default: 85)')
    parser.add_argument('--eta_min', type=float, default=1e-9,
                        help='minimum learning rate for CosineAnnealingLR (default: 1e-9)')
    #parser.add_argument('--lr_decay_interval', type=int, default=100,
    #                    help='decay learning rate every N epochs(default: 100)')
    
    # 新增熱重啟相關參數
    parser.add_argument('--t_0', type=int, default=25,
                        help='Number of epochs for the first restart cycle (T_0 for CosineAnnealingWarmRestarts)')
    parser.add_argument('--t_mult', type=int, default=1,
                        help='A factor to increase T_i after a restart (T_mult for CosineAnnealingWarmRestarts)')
  
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='start epoch of training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='training batch size (default: 8)')#原為16
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--wandb_key', type=str, default=None, help='Weights & Biases API Key')
    parser.add_argument('--wandb_project', type=str, default='hdr-transformer', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (user or team)')
    parser.add_argument('--wandb_run_name', type=str, default=f"run-{time.strftime('%Y%m%d-%H%M')}", help='Custom run name for W&B')
    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    for batch_idx, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                                             batch_data['input2'].to(device)
        label = batch_data['label'].to(device)
        #數據驗證
        assert not torch.isnan(batch_ldr0).any() and not torch.isinf(batch_ldr0).any(), f"Epoch {epoch}, Batch {batch_idx}: NaN/inf in ldr0"
        assert not torch.isnan(batch_ldr1).any() and not torch.isinf(batch_ldr1).any(), f"Epoch {epoch}, Batch {batch_idx}: NaN/inf in ldr1"
        assert not torch.isnan(batch_ldr2).any() and not torch.isinf(batch_ldr2).any(), f"Epoch {epoch}, Batch {batch_idx}: NaN/inf in ldr2"
        assert not torch.isnan(label).any() and not torch.isinf(label).any(), f"Epoch {epoch}, Batch {batch_idx}: NaN/inf in label"

        #檢查標籤範圍是否適用於 range_compressor (mu=5000)
        min_label_val = torch.min(label)
        mu = 5000 # 假設 mu 值
        required_min = -1.0 / mu
        assert min_label_val > required_min, f"Epoch {epoch}, Batch {batch_idx}: Label contains values <= -1/mu ({required_min}). Min label: {min_label_val}"
        

        pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
        loss = criterion(pred, label)
        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        # 添加梯度裁剪 (max_norm=1.0 是一個常用起始值，可根據需要調整)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:3f})'.format(
                epoch,
                batch_idx * args.batch_size,
                len(train_loader.dataset),
                100. * batch_idx * args.batch_size / len(train_loader.dataset),
                loss.item(),
                batch_time=batch_time,
                data_time=data_time
            ))
            step = epoch * len(train_loader) + batch_idx
            wandb.log({"batch_train_loss": loss.item()}, step=step)
    print(f'Training Epoch {epoch}: Average Loss: {losses.avg:.4f}')
    return losses.avg  # 返回平均 loss

# for evaluation with limited GPU memory
def test_single_img(model, img_dataset, device):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=1, shuffle=False)
    # model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader, total=len(dataloader)):
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), \
                                                 batch_data['input1'].to(device), \
                                                 batch_data['input2'].to(device)
            output = model(batch_ldr0, batch_ldr1, batch_ldr2)
            img_dataset.update_result(torch.squeeze(output.detach().cpu()).numpy().astype(np.float32))
    pred, label = img_dataset.rebuild_result()
    return pred, label

def test(args, model, device, scheduler, optimizer, epoch, cur_psnr, criterion=None, **kwargs):
    model.eval()
    test_datasets = SIG17_Test_Dataset(args.dataset_dir, args.patch_size) 
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    val_loss = AverageMeter()  # 新增一個 AverageMeter 來追蹤 validation loss
    
    with torch.no_grad():
        for idx, img_dataset in enumerate(test_datasets):
            pred_img, label = test_single_img(model, img_dataset, device)
            
            # 如果提供了 criterion，計算 validation loss
            if criterion is not None:
                # 將 numpy 數組轉換為 tensor
                pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).to(device)
                label_tensor = torch.from_numpy(label).unsqueeze(0).to(device)
                loss = criterion(pred_tensor, label_tensor)
                val_loss.update(loss.item())
            if epoch == args.start_epoch and idx == 0:
                # 將 LDR (input) 和 HDR (prediction/label) 轉成 wandb 可顯示的格式
                # 假設 ldr (input0) 已經是 [0,1] 範圍的 RGB 圖像
                ldr_img = img_dataset.ldr_images[0].transpose(1, 0, 2)
                # 對 HDR 圖像進行色調映射 (tonemapping) 以便於顯示
                pred_tonemapped = np.clip(tonemap(pred_img.transpose(1, 2, 0)), 0, 1)
                label_tonemapped = np.clip(tonemap(label.transpose(1, 2, 0)), 0, 1)

                wandb.log({
                    "examples": [
                        wandb.Image(ldr_img, caption="Input LDR"),
                        wandb.Image(pred_tonemapped, caption=f"Prediction (Epoch {epoch})"),
                        wandb.Image(label_tonemapped, caption="Ground Truth")
                    ]
                }, step=epoch)

            # 計算其他指標，如 PSNR 和 SSIM
            scene_psnr_l = compare_psnr(label, pred_img, data_range=1.0)
            label_mu = range_compressor(label)
            pred_img_mu = range_compressor(pred_img)
            scene_psnr_mu = compare_psnr(label_mu, pred_img_mu, data_range=1.0)
            
            # 準備圖像用於 SSIM 計算
            pred_img_for_ssim = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
            label_for_ssim = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
            pred_img_mu_for_ssim = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
            label_mu_for_ssim = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
            
            scene_ssim_l = calculate_ssim(pred_img_for_ssim, label_for_ssim)
            scene_ssim_mu = calculate_ssim(pred_img_mu_for_ssim, label_mu_for_ssim)
            
            psnr_l.update(scene_psnr_l)
            ssim_l.update(scene_ssim_l)
            psnr_mu.update(scene_psnr_mu)
            ssim_mu.update(scene_ssim_mu)
    
    # 在輸出中增加 loss 資訊
    if criterion is not None:
        print('==Validation==\tLoss: {:.4f}\tPSNR_l: {:.4f}\t PSNR_mu: {:.4f}\t SSIM_l: {:.4f}\t SSIM_mu: {:.4f}'.format(
            val_loss.avg,
            psnr_l.avg,
            psnr_mu.avg,
            ssim_l.avg,
            ssim_mu.avg
        ))
    else:
        print('==Validation==\tPSNR_l: {:.4f}\t PSNR_mu: {:.4f}\t SSIM_l: {:.4f}\t SSIM_mu: {:.4f}'.format(
            psnr_l.avg,
            psnr_mu.avg,
            ssim_l.avg,
            ssim_mu.avg
        ))

    # save_model
    save_dict = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_dict, os.path.join(args.logdir, 'val_latest_checkpoint.pth'))
    if psnr_mu.avg > cur_psnr[0]:
        best_checkpoint_path = os.path.join(args.logdir, 'best_checkpoint.pth')
        torch.save(save_dict, best_checkpoint_path)
        cur_psnr[0] = psnr_mu.avg
        with open(os.path.join(args.logdir, 'best_checkpoint.json'), 'w') as f:
            f.write('best epoch:' + str(epoch) + '\n')
            f.write('Validation set: Average PSNR: {:.4f}, PSNR_mu: {:.4f}, SSIM_l: {:.4f}, SSIM_mu: {:.4f}\n'.format(
                psnr_l.avg,
                psnr_mu.avg,
                ssim_l.avg,
                ssim_mu.avg
                ))
        artifact = wandb.Artifact(f'model-best-{wandb.run.id}', type='model')
        artifact.add_file(best_checkpoint_path)
        wandb.log_artifact(artifact, aliases=['best'])

    # W&B: 7. 記錄所有驗證指標
    metrics_to_log = {
        "Validation/val_loss": val_loss.avg ,
        "Validation/val_psnr_l": psnr_l.avg,
        "Validation/val_psnr_mu": psnr_mu.avg,
        "Validation/val_ssim_l": ssim_l.avg,
        "Validation/val_ssim_mu": ssim_mu.avg,
        "epoch": epoch
    }
    if criterion is not None:
         metrics_to_log["Validation/val_loss"] = val_loss.avg

    wandb.log(metrics_to_log)
    return val_loss.avg if criterion is not None else 0.0, psnr_l.avg, psnr_mu.avg

def main():
    # settings
    args = get_args()
    if args.wandb_key:
        wandb.login(key=args.wandb_key)

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=args, # 儲存所有超參數
    )
    # W&B: Define custom x-axis for epoch-level metrics
    # This tells wandb to use 'epoch' as the x-axis for these metrics
    wandb.define_metric("epoch")
    wandb.define_metric("avg_train_loss", step_metric="epoch")
    wandb.define_metric("learning_rate", step_metric="epoch")
    wandb.define_metric("Validation/*", step_metric="epoch")
    wandb.define_metric("examples", step_metric="epoch")
    train_losses = []
    val_losses = []
    val_psnrs = []
    val_mu_psnrs = []
    # random seed
    if args.seed is not None:
        set_random_seed(args.seed)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # model architectures
    model = HDRTransformer(embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6], 
                           mlp_ratio=2, in_chans=3,
                           use_dynamic_tanh=args.use_dynamic_tanh,
                           dyt_alpha_init_value=args.dyt_alpha_init_value)
    cur_psnr = [-1.0]
    # init
    if args.init_weights:
        init_parameters(model)
    # loss
    loss_dict = {
        0: L1MuLoss,
        1: JointReconPerceptualLoss,
        }
    criterion = loss_dict[args.loss_func]().to(device)
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                  eps=1e-08, weight_decay=args.weight_decay) # 修改為 AdamW
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t_0, T_mult=args.t_mult, eta_min=args.eta_min)
    #檢查DyT
    if args.use_dynamic_tanh:
        print("="*50)
        print("Model structure after DyT conversion (printed from training script):")
        print(model) # 如果轉換是在模型內部 __init__ 呼叫的，這裡會顯示轉換後的結構
        print("="*50)
    model.to(device)
    wandb.watch(model, criterion, log="all", log_freq=args.log_interval)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if args.resume:
        if os.path.isfile(args.resume):
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
             # 為了向後相容，檢查 checkpoint 中是否有 scheduler 狀態
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("===> Loaded scheduler state.")
            else:
                # 如果舊的 checkpoint 沒有 scheduler 狀態，可以根據 start_epoch 手動更新
                print("===> No scheduler state found in checkpoint. Manually updating scheduler.")
                for _ in range(args.start_epoch - 1):
                    scheduler.step()
            print("===> Loaded checkpoint: epoch {}".format(checkpoint['epoch']))
        else:
            print("===> No checkpoint is founded at {}.".format(args.resume))
    
    # dataset and dataloader
    train_dataset = SIG17_Training_Dataset(root_dir=args.dataset_dir, sub_set=args.sub_set, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    dataset_size = len(train_loader.dataset)
    loss_name = criterion.__class__.__name__
    # 建立一個更有組織、更詳細的啟動訊息
    print(f'''
+-----------------------------------------------------------------+
|               🚀 開始訓練 HDR-Transformer 🚀                  |
+-----------------------------------------------------------------+
| [日誌與檢查點資訊]                                                |
|   W&B Project:       {args.wandb_project}
|   W&B Run Name:      {args.wandb_run_name}
|   從檢查點恢復:      {args.resume if args.resume else "否"}
|   檢查點儲存路徑:    {args.logdir}
|                                                                 |
| [資料集與讀取器資訊]                                              |
|   資料集路徑:        {args.dataset_dir}
|   子資料夾:          {args.sub_set}
|   裁切尺寸:          {args.patch_size}
|   訓練樣本數:        {dataset_size}
|   批次大小 (Batch):   {args.batch_size}
|   工作執行緒數:      {args.num_workers}
|                                                                 |
| [模型與訓練設定]                                                  |
|   總訓練週期 (Epochs): {args.epochs}
|   隨機總子:          {args.seed}
|   使用設備:          {device.type.upper()}
|   使用 DynamicTanh:  {"是" if args.use_dynamic_tanh else "否"}
|                                                                 |
| [優化器與學習率排程]                                              |
|   優化器:            AdamW
|   損失函數:          {loss_name} (ID: {args.loss_func})
|   初始學習率:        {args.lr}
|   權重衰減 (WD):     {args.weight_decay}
|   學習率排程器:      CosineAnnealingWarmRestarts 
|   排程器 T_0:        {args.t_0}                    
|   排程器 T_mult:     {args.t_mult}                    
|   排程器 eta_min:    {args.eta_min}
+-----------------------------------------------------------------+
''')

    for epoch in range(args.start_epoch, args.epochs + 1):
        # 更新學習率
        #adjust_learning_rate(args, optimizer, epoch)
        
        # 訓練階段
        train_loss = train(args, model, device, train_loader, optimizer, epoch, criterion)
        train_losses.append(train_loss)
        wandb.log({"avg_train_loss": train_loss, "epoch": epoch})
        # 驗證階段
        val_loss, val_psnr, val_mu_psnr = test(args, model, device, optimizer, scheduler, epoch, cur_psnr, criterion=criterion)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_mu_psnrs.append(val_mu_psnr)

        # W&B: Log learning rate for the epoch
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr, "epoch": epoch})

        scheduler.step()
    wandb.finish() 

if __name__ == '__main__':
    # 輔助函數，用於在 wandb 中顯示圖像 (需要 numpy 和 skimage)
    try:
        from skimage.transform import rescale
        # 簡單的 Reinhard tonemapping
        def tonemap(hdr_image, gamma=2.2):
            return np.power(hdr_image / (hdr_image + 1), 1/gamma)
    except ImportError:
        print("Warning: skimage not installed. Image logging will be disabled.")
        def tonemap(hdr_image): return hdr_image
    main()