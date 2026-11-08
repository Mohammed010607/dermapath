"""Segmentation model implementations and a config-driven factory."""

from pathlib import Path

from .unet import UNet

__all__ = ["UNet", "build_model", "load_pretrained"]


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


def load_pretrained(model, config) -> bool:
    """Load ImageNet pretrained weights into a model, based on its config.

    UNet has no pretrained initialisation (it is the from-scratch baseline). TransUNet and
    Swin-UNet load the ImageNet weights named by ``model.pretrained`` if that file exists.
    Returns True if weights were loaded.
    """
    model_cfg = config["model"]
    name = model_cfg["name"].lower()
    path = model_cfg.get("pretrained")
    if not path:
        return False
    path = Path(path)
    if not path.exists():
        print(f"pretrained weights not found ({path}); using random initialisation")
        return False

    if name == "transunet":
        import numpy as np

        model.load_from(np.load(str(path)))
    elif name == "swin_unet":
        from .swin_unet import load_swin_pretrained

        load_swin_pretrained(model, path)
    else:
        print(f"no pretrained loader for {name!r}; using random initialisation")
        return False

    print(f"loaded pretrained weights: {path}")
    return True
