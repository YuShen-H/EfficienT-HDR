import torch.nn as nn
from mmpretrain.models.utils.inverted_residual import InvertedResidual

def BuildInvertedResidual(in_channels, mid_channels, out_channels, kernel_size=3, stride=1, wise=True, with_res_shortcut=None):
    """
    建立一個 InvertedResidual 模組。
    這是 MobileNetV2/V3 中的核心 CNN 區塊。
    """
    se_cfg = None
    if wise:
        se_cfg = dict(
            channels=mid_channels,
            ratio=4,
            act_cfg=(dict(type='ReLU'),
                     dict(
                         type='HSigmoid',
                         bias=3,
                         divisor=6,
                         min_value=0,
                         max_value=1)))
    act = 'HSwish'
    tmp = InvertedResidual(
        in_channels=in_channels,
        out_channels=out_channels,
        mid_channels=mid_channels,
        kernel_size=kernel_size,
        stride=stride,
        se_cfg=se_cfg,
        act_cfg=dict(type=act),
    )
    if with_res_shortcut is not None:
        tmp.with_res_shortcut = with_res_shortcut
    return tmp