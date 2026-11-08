"""Segmentation model implementations and a config-driven factory."""

from .unet import UNet

__all__ = ["UNet", "build_model"]


def build_model(config: dict):
    """Instantiate a model from the ``model`` section of a loaded config."""
    model_cfg = config["model"]
    name = model_cfg["name"].lower()

    if name == "unet":
        return UNet(
            in_channels=model_cfg.get("in_channels", 3),
            out_channels=model_cfg.get("out_channels", 1),
            base_channels=model_cfg.get("base_channels", 64),
            bilinear=model_cfg.get("bilinear", False),
        )

    if name == "transunet":
        from .transunet import build_transunet

        return build_transunet(
            img_size=model_cfg.get("img_size", 224),
            num_classes=model_cfg.get("out_channels", 1),
            vit_name=model_cfg.get("vit_name", "R50-ViT-B_16"),
        )

    if name == "swin_unet":
        from .swin_unet import build_swin_unet

        return build_swin_unet(
            img_size=model_cfg.get("img_size", 224),
            num_classes=model_cfg.get("out_channels", 1),
        )

    raise ValueError(f"unknown model name: {name!r}")
