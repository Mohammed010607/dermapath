"""Swin-UNet (pure transformer, Swin-Tiny), adapted from the official implementation.

``swin_transformer_unet_skip_expand_decoder_sys.py`` is the official Swin-Unet source
(https://github.com/HuCaoFighting/Swin-Unet; see ATTRIBUTION.md - the repository ships no
licence file, and the code derives from Microsoft's MIT-licensed Swin Transformer). Only
this builder is project code: it instantiates the Swin-Tiny variant directly (bypassing the
repository's yacs config layer) for binary lesion segmentation at 224x224.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

# Swin-Tiny hyperparameters (the file's default depths [2,2,2,2] is NOT Swin-T).
SWIN_TINY = dict(
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
)


def build_swin_unet(img_size: int = 224, num_classes: int = 1,
                    drop_path_rate: float = 0.1) -> SwinTransformerSys:
    """Build Swin-UNet (Swin-Tiny encoder/decoder) for binary segmentation.

    The decoder depths are left at the implementation default ([1, 2, 2, 2]), matching the
    official SwinUnet wrapper, which does not override them.
    """
    return SwinTransformerSys(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=SWIN_TINY["embed_dim"],
        depths=SWIN_TINY["depths"],
        num_heads=SWIN_TINY["num_heads"],
        window_size=SWIN_TINY["window_size"],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        final_upsample="expand_first",
    )


def load_swin_pretrained(model: SwinTransformerSys, ckpt_path) -> str:
    """Load ImageNet Swin-Tiny weights into both encoder and decoder.

    Adapted from the official Swin-Unet ``load_from``: the encoder ``layers.*`` weights are
    copied and also mirrored into the decoder ``layers_up.*`` (with reversed stage order),
    since pretraining both halves is important for a pure-transformer model. Weights whose
    shapes do not match (e.g. the classification head) are dropped. Returns the
    missing/unexpected-key report from ``load_state_dict``.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pretrained = checkpoint["model"] if "model" in checkpoint else checkpoint

    model_dict = model.state_dict()
    full = copy.deepcopy(pretrained)
    for k, v in pretrained.items():
        if "layers." in k:
            stage = 3 - int(k[7:8])  # mirror encoder stage onto the decoder
            full["layers_up." + str(stage) + k[8:]] = v
    for k in list(full.keys()):
        if k in model_dict and full[k].shape != model_dict[k].shape:
            del full[k]

    msg = model.load_state_dict(full, strict=False)
    return str(msg)


__all__ = ["build_swin_unet", "load_swin_pretrained", "SwinTransformerSys", "SWIN_TINY"]
