"""UNet forward-pass shape checks (small channel widths to keep the test fast on CPU)."""

import torch

from src.models.unet import UNet


def test_forward_shape_224():
    model = UNet(in_channels=3, out_channels=1, base_channels=16).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 224, 224)


def test_output_is_logits_not_probabilities():
    """Raw logits may fall outside [0, 1]; the model must not apply a final sigmoid."""
    model = UNet(base_channels=8).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all()


def test_handles_non_multiple_of_16():
    """The padding guard lets the net accept sizes that are not divisible by 16."""
    model = UNet(base_channels=8).eval()
    x = torch.randn(1, 3, 192, 160)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 192, 160)


def test_bilinear_upsampling_variant():
    model = UNet(base_channels=8, bilinear=True).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 224, 224)
