#-*- coding:utf-8 -*-  
import math
import torch
import torch.nn as nn
import torchvision
import numpy as np

def range_compressor(hdr_img, mu=5000):
    # --- 新增/修改開始 ---
    # 選項 1: 鉗位輸入 hdr_img (確保輸入嚴格 > -1/mu)
    # 加上一個非常小的數值避免邊界問題
    min_val = -1.0 / mu + 1e-7 
    hdr_img_clamped = torch.clamp(hdr_img, min=min_val)
    log_arg = 1 + mu * hdr_img_clamped

    # 再次鉗位 log_arg 以防因浮點數精度問題導致其恰好為 0 或負數
    log_arg_clamped = torch.clamp(log_arg, min=1e-9) 

    return torch.log(log_arg_clamped) / math.log(1 + mu)
    # --- 新增/修改結束 ---

    # # 原始程式碼:
    # return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu) 


class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super(L1MuLoss, self).__init__()
        self.mu = mu

    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)
        mu_label = range_compressor(label, self.mu)
        return nn.L1Loss()(mu_pred, mu_label)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class JointReconPerceptualLoss(nn.Module):
    def __init__(self, alpha=0.01, mu=5000, resize_inputs=True):
        super(JointReconPerceptualLoss, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = L1MuLoss(self.mu)
        self.loss_vgg = VGGPerceptualLoss(resize=True)  # 強制使用 resize=True 以處理 VGG 需求
        self.resize_inputs = resize_inputs  # 是否在計算損失前調整尺寸
        
    def forward(self, input, target):
        # 檢查並調整尺寸不匹配
        if self.resize_inputs and (input.shape[2] != target.shape[2] or input.shape[3] != target.shape[3]):
            # 將輸入調整為目標尺寸（假設目標尺寸是預期的）
            input = torch.nn.functional.interpolate(
                input, 
                size=(target.shape[2], target.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
            print(f"已將輸入 {input.shape} 調整為目標尺寸 {target.shape}")
        
        # 進行範圍壓縮
        input_mu = range_compressor(input, self.mu)
        target_mu = range_compressor(target, self.mu)
        
        # 計算損失
        loss_recon = self.loss_recon(input, target)
        loss_vgg = self.loss_vgg(input_mu, target_mu)
        loss = loss_recon + self.alpha * loss_vgg
        
        return loss