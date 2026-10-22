# EfficienT-HDR

### An Efficient Transformer-Based Framework via Multi-Exposure Fusion for HDR Reconstruction

Yu-Shen Huang, Tzu-Han Chen, Cheng-Yen Hsiao, and Shaou-Gang Miaou

[Paper](https://arxiv.org/abs/2509.19779)

## Overview

EfficienT-HDR is a lightweight Transformer-based framework for
HDR reconstruction from multiple LDR images captured at different
exposures. It aims to reduce computational cost while suppressing
ghosting artifacts caused by motion between exposures.

Built upon HDR-Transformer, this work explores efficient feature
extraction and adaptive fusion to balance reconstruction quality
and computational efficiency.

## Method Highlights

- **Inverted Residual Embedding (IRE):**
  Efficient feature extraction with reduced computational overhead.
- **Intersection-Aware Adaptive Fusion (IAAF):**
  Adaptive fusion to mitigate ghosting across exposures.
- **Dynamic Tanh (DyT):**
  An alternative to Layer Normalization in Transformer blocks.
- **Enhanced Multi-Scale Dilated Convolution (E-MSDC):**
  Multi-scale local feature extraction.

The paper presents two variants: EfficienT-HDR and EfficienT-HDR-Lite,
targeting different quality–efficiency tradeoffs.

## Repository Status

This repository is being organized for public release.
The mapping between experimental model versions and the paper's
final variants, pretrained checkpoints, and reproducible execution
instructions is being consolidated.

## Citation

If you find this work useful, please cite:

@article{huang2025efficienthdr,
  title={EfficienT-HDR: An Efficient Transformer-Based Framework
         via Multi-Exposure Fusion for HDR Reconstruction},
  author={Huang, Yu-Shen and Chen, Tzu-Han and
          Hsiao, Cheng-Yen and Miaou, Shaou-Gang},
  journal={arXiv preprint arXiv:2509.19779},
  year={2025}
}

## Acknowledgments

This project builds upon the PyTorch implementation of
[HDR-Transformer](https://github.com/ljzycmd/HDR-Transformer).
We thank the original authors for making their code available.

This research was supported by Taiwan's National Science and
Technology Council (NSTC), under the College Student Research
Scholarship, grant 114-2813-C-033-037-E.
