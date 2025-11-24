#-*- coding:utf-8 -*-
#YCbCr+MSGDC+Dyt
import math
import time
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from .mmobj import BuildInvertedResidual 

from timm.models.layers import DropPath, to_2tuple, trunc_normal_, LayerNorm2d
from thop import profile



#add y
def _rgb_to_y(r: Tensor, g: Tensor, b: Tensor) -> Tensor:
    y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return y
#cb
def _rgb_to_cb(r: Tensor, g: Tensor, b: Tensor) -> Tensor:
    cb: Tensor = -0.168736 * r - 0.331264 * g + 0.5 * b
    return cb
#cr
def _rgb_to_cr(r: Tensor, g: Tensor, b: Tensor) -> Tensor:
    cr: Tensor = 0.5 * r - 0.418688 * g - 0.081312 * b
    return cr



def rgb_to_y(image: Tensor) -> Tensor:
    r"""Convert an RGB image to Y.

    Args:
        image: RGB Image to be converted to Y with shape :math:`(*, 3, H, W)`.

    Returns:
        Y version of the image with shape :math:`(*, 1, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_y(input)  # 2x1x4x5
    """
    # if not isinstance(image, Tensor):
    #     raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    # if len(image.shape) < 3 or image.shape[-3] != 3:
    #     raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]

    y: Tensor = _rgb_to_y(r, g, b)
    return y

def rgb_to_cb(image: Tensor) -> Tensor:
    r"""Convert an RGB image to Cb.

    Args:
        image: RGB Image with shape (*, 3, H, W)

    Returns:
        Cb channel with shape (*, 1, H, W)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must be (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]

    cb: Tensor = _rgb_to_cb(r, g, b)
    return cb

def rgb_to_cr(image: Tensor) -> Tensor:
    r"""Convert an RGB image to Cr.

    Args:
        image: RGB Image with shape (*, 3, H, W)

    Returns:
        Cr channel with shape (*, 1, H, W)
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must be (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]

    cr: Tensor = _rgb_to_cr(r, g, b)
    return cr

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), 
        stride=stride,
        bias=bias)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# MSGDC Block    
class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn,1,1]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class MSGDC(nn.Module):
    def __init__(self, in_chn, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding='same'):
        super(MSGDC, self).__init__()

        self.move = LearnableBiasnn(in_chn)

        # 將 'padding' 參數換成計算出的具體數值
        #self.cov1 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding=1, dilation=dilation1, groups=4, bias=True)
        #self.cov2 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding=3, dilation=dilation2, groups=4, bias=True)
        #self.cov3 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding=5, dilation=dilation3, groups=4, bias=True)

        self.cov1 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding,dilation1,4,bias=True)
        self.cov2 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding,dilation2,4,bias=True)
        self.cov3 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding,dilation3,4,bias=True)

        self.norm = nn.LayerNorm(in_chn, eps=1e-6)

        self.act1 = nn.GELU()
        self.act2 = nn.GELU() 
        self.act3 = nn.GELU()

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.move(x)
        x1 = self.cov1(x).permute(0, 2, 3, 1).flatten(1,2)
        x1 = self.act1(x1)
        x2 = self.cov2(x).permute(0, 2, 3, 1).flatten(1,2)
        x2 = self.act2(x2)
        x3 = self.cov3(x).permute(0, 2, 3, 1).flatten(1,2)
        x3 = self.act3(x3)
        x = self.norm(x1+x2+x3)
        return x.permute(0, 2, 1).view(-1, C, H, W).contiguous()

#DyT
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        #self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
        #self.bias = nn.Parameter(torch.zeros(*self.normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


def convert_ln_to_dyt(module, alpha_init_value=0.5):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d), alpha_init_value=alpha_init_value)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    


class LocalContextExtractor(nn.Module):

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class ContextAwareTransformer(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.lce = LocalContextExtractor(self.dim)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # local context features
        lcf = x.permute(0, 3, 1, 2)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # local context
        lc = self.lce(lcf)
        lc = lc.view(B, C, H * W).permute(0, 2, 1)
        x = lc + x

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

   


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            ContextAwareTransformer(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size) # B L C
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    


class ContextAwareTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.dilated_conv = MSGDC(dim)
        elif resi_connection == '3conv':
             self.dilated_conv = MSGDC(dim, dilation1=1, dilation2=3, dilation3=5)
            # to save parameters and memory
            # self.dilated_conv = nn.Sequential(
            #     nn.Conv2d(dim, dim // 4, kernel_size=3, padding=2, dilation=2), 
            #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #     nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #     nn.Conv2d(dim // 4, dim, kernel_size=3, padding=2, dilation=2)
            #     )

        self.patch_embed = PatchEmbed(
            img_size=input_resolution[0], patch_size=1, in_chans=dim, embed_dim=dim,
            norm_layer=None, stride=1)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        res = self.residual_group(x, x_size) # B L C
        res_unembed = self.patch_unembed(res, x_size) # B C H W # Use different variable name
        res_conv = self.dilated_conv(res_unembed) # Use res_unembed
        res_embed = self.patch_embed(res_conv) + x # Use res_conv
        return res_embed

    


class PatchEmbed(nn.Module):
    """
    MODIFIED: 2D Image to Patch Embedding.
    Now uses an InvertedResidual block for more powerful feature extraction
    and spatial downsampling via striding.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=60, embed_dim=60, norm_layer=None, stride=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        
        # ==================== 核心修改點 ====================
        # 1. 確保 stride 總是一個元組（例如，傳入的 2 會變成 (2, 2)）
        stride = to_2tuple(stride)
        
        # 2. 統一使用索引 [0] 和 [1] 來獲取高和寬的 stride
        self.patches_resolution = [
            int(img_size[0] // stride[0]), 
            int(img_size[1] // stride[1])
        ]
        # ====================================================

        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.img_size = img_size
        self.patch_size = to_2tuple(patch_size) # 保持為元組
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        mid_chans = in_chans * 4
        self.proj = BuildInvertedResidual(
            in_channels=in_chans,
            mid_channels=mid_chans,
            out_channels=embed_dim,
            kernel_size=3,
            stride=stride[0] # BuildInvertedResidual 通常接受一個整數 stride
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_chans, f"Input channels ({C}) doesn't match in_chans ({self.in_chans})"

        x = self.proj(x)
        
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

   

class SpatialAttentionModule(nn.Module):

    def __init__(self, dim):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map

class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1,
            stride=[1, 1]
            ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias, stride=stride[i]))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        if x.size() != res.size():
            res += F.interpolate(x, size=res.size()[2:], mode='bilinear', align_corners=False)
        else:
            res += x

        return res

class IAAF(nn.Module):
    def __init__(self, in_channel = 6, out_channel = 3, n_channels=16, act=nn.ReLU(True),*args, **kwargs) -> None:
        super(IAAF, self).__init__()
        # 這些卷積層共同構成了學習「交集」的網路
        self.conv1_post_g = nn.Conv2d(in_channels=in_channel, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_post_g = ResBlock(conv=default_conv, n_feats=n_channels, kernel_size=3, act=act)
        self.conv3_post_g = nn.Conv2d(in_channels=n_channels, out_channels=out_channel, kernel_size=1, stride=1, padding=0)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, gamma, llie, x=None, get_all=False,*args, **kwarg): 
        
        base_img = x if x is not None else llie
        # 1. 計算 R^gamma
        out_g = torch.pow(base_img, gamma)

        # 2. 學習交集 ∩(R, R^gamma)
        # 首先將兩個特徵圖串聯 (concatenate)
        out = self.conv1_post_g(torch.cat((out_g, llie), dim=1))
        out = self.conv2_post_g(out)
        # 最終得到學習出的交集
        intersection = self.conv3_post_g(out)
        
        # 3. 實現最終融合公式
        output = out_g + llie - intersection
        
        output = torch.clamp(output, min=1e-4, max=1)
        if get_all:
            return output, out_g, intersection
        return output, out_g

class HDRTransformer(nn.Module):

    def __init__(self, img_size=128, patch_size=1, in_chans=3,
                 embed_dim=60, depths=[6, 6, 6], num_heads=[6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv',
                 use_dynamic_tanh=True,
                 dyt_alpha_init_value=0.5,
                 **kwargs):
        super(HDRTransformer, self).__init__()
        
        #num_in_ch = in_chans
        num_out_ch = 3
        ################################### 1. Feature Extraction Network ###################################
        # coarse feature
        self.conv_f1 = nn.Conv2d(3, embed_dim, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(3, embed_dim, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(3, embed_dim, 3, 1, 1)
        # spatial attention module
        iaaf_n_channels = 16 # 這是一個超參數，可以設為 16 或 32
        self.iaaf_high = IAAF(in_channel=embed_dim * 2, out_channel=embed_dim, n_channels=iaaf_n_channels)
        self.iaaf_low  = IAAF(in_channel=embed_dim * 2, out_channel=embed_dim, n_channels=iaaf_n_channels)
        #self.att_module_l = SpatialAttentionModule(embed_dim)
        #self.att_module_h = SpatialAttentionModule(embed_dim)
        self.conv_first = nn.Conv2d(embed_dim * 3, embed_dim, 3, 1, 1)
        ################################### 2. HDR Reconstruction Network ###################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, stride=2)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        num_patches = self.patch_embed.num_patches

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.absolute_pos_embed = None # Ensure it's defined even if not used

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Context-aware Transformer Blocks (CTBs)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ContextAwareTransformerBlock(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)
        if use_dynamic_tanh:
            #print(f"Applying DynamicTanh with alpha_init_value={dyt_alpha_init_value} to HDRTransformer...")
            convert_ln_to_dyt(self, alpha_init_value=dyt_alpha_init_value) # 呼叫轉換函數
            #print("="*50)
            #print("HDRTransformer structure after DyT conversion:")
            # print(self) # 可以取消註解來檢查結構
            #print("="*50)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # x [B, embed_dim, h, w]
        x_size = self.patch_embed.patches_resolution
        x = self.patch_embed(x) # B L C
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x1, x2, x3):
        
        x1_rgb = x1[:, :3, :, :]
        x2_rgb = x2[:, :3, :, :]
        x3_rgb = x3[:, :3, :, :]
    
    # YCbCr 轉換
        y1 = rgb_to_y(x1_rgb)
        cb1 = rgb_to_cb(x1_rgb)
        cr1 = rgb_to_cr(x1_rgb)
        x1_ycbcr = torch.cat((y1, cb1, cr1), dim=1)

        y2 = rgb_to_y(x2_rgb)
        cb2 = rgb_to_cb(x2_rgb)
        cr2 = rgb_to_cr(x2_rgb)
        x2_ycbcr = torch.cat((y2, cb2, cr2), dim=1)

        y3 = rgb_to_y(x3_rgb)
        cb3 = rgb_to_cb(x3_rgb)
        cr3 = rgb_to_cr(x3_rgb)
        x3_ycbcr = torch.cat((y3, cb3, cr3), dim=1)

        # feature extraction network
        # coarse feature
        f1 = self.conv_f1(x1_ycbcr)
        f2 = self.conv_f2(x2_ycbcr)
        f3 = self.conv_f3(x3_ycbcr)

        # spatial feature attention 
        gamma_tensor = torch.tensor([1.0], device=f1.device)
        f1_aligned, _ = self.iaaf_high(gamma=gamma_tensor, llie=f1, x=f2)

    # 通道二：對齊欠曝幀 f3，提取亮部細節
        f3_aligned, _ = self.iaaf_low(gamma=gamma_tensor, llie=f3, x=f2)

    # 最終融合：將對齊後的特徵與錨點 f2 串聯
        x = self.conv_first(torch.cat((f1_aligned, f2, f3_aligned), dim=1))
        #f1_att_m = self.att_module_h(f1, f2)
        #f1_att = f1 * f1_att_m
        #f3_att_m = self.att_module_l(f3, f2)
        #f3_att = f3 * f3_att_m
        #x = self.conv_first(torch.cat((f1_att, f2, f3_att), dim=1))

        # CTBs for HDR reconstruction
        # The forward_features part now processes features derived from YCbCr
        features_out = self.forward_features(x)
        features_out_upsampled = torch.nn.functional.interpolate(
        features_out,
        size=(x.shape[2], x.shape[3]), # 目標尺寸為 x 的高和寬
        mode='bilinear',
        align_corners=False
    )
        res = self.conv_after_body(features_out_upsampled + x) # Apply residual connection

        # The final output combines the middle frame's initial features (f2, derived from YCbCr)
        # with the deep features (res). The network learns to map this back to RGB.
        x = self.conv_last(f2 + res)
        x = torch.sigmoid(x) # Output is assumed to be the final RGB image
        return x














