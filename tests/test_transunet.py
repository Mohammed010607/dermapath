"""TransUNet forward-pass shape check."""

import torch

from src.models import build_model
from src.utils.config import load_config


def test_transunet_forward_shape():
    cfg = load_config("configs/transunet.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 224, 224)
