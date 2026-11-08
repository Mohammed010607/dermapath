"""TransUNet (CNN + ViT hybrid), adapted from the official implementation.

The model code in this package - ``vit_seg_modeling.py``, ``vit_seg_configs.py``,
``vit_seg_modeling_resnet_skip.py`` - is the official TransUNet source
(https://github.com/Beckschen/TransUNet, Apache 2.0; see the LICENSE file kept alongside).
Only this thin builder is project code: it configures the R50-ViT-B_16 variant for binary
lesion segmentation at 224x224 and exposes it through the shared training harness.
"""

from __future__ import annotations

import copy

from .vit_seg_modeling import CONFIGS, VisionTransformer


def build_transunet(img_size: int = 224, num_classes: int = 1,
                    vit_name: str = "R50-ViT-B_16", patch_size: int = 16) -> VisionTransformer:
    """Build TransUNet for binary segmentation.

    The segmentation head's output channels come from ``config.n_classes`` and the ViT patch
    grid must match the ResNet feature-map size (img_size / 16). The config is deep-copied so
    the module-level ``CONFIGS`` entry is not mutated.
    """
    config = copy.deepcopy(CONFIGS[vit_name])
    config.n_classes = num_classes
    config.patches.grid = (img_size // patch_size, img_size // patch_size)
    return VisionTransformer(config, img_size=img_size, num_classes=num_classes)


__all__ = ["build_transunet", "VisionTransformer", "CONFIGS"]
