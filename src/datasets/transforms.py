"""Shared preprocessing and augmentation for every model in the comparison.

A single transform definition is used by all three architectures so that the only thing
that differs between experiments is the network. Both pipelines resize to 224x224 and
normalise with ImageNet statistics; the training pipeline adds augmentation (flips,
rotation, colour jitter, elastic deformation), which is NEVER applied to validation or
test data.

Everything is expressed as MONAI dictionary transforms keyed ``"image"`` and ``"label"``.
Geometric transforms use bilinear interpolation for the image and nearest-neighbour for the
mask, so the mask stays strictly binary ({0, 1}).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    MapTransform,
    NormalizeIntensityd,
    Rand2DElasticd,
    RandFlipd,
    RandomizableTransform,
    RandRotated,
    Resized,
)
from torchvision.transforms import functional as TF

# ImageNet statistics - the pretrained ViT / Swin-Tiny encoders expect this normalisation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


class RandColorJitterd(MapTransform, RandomizableTransform):
    """Colour jitter on the image only, driven by MONAI's random state.

    Brightness, contrast, saturation, and hue factors are drawn from MONAI's RNG (so the
    augmentation is reproducible with the shared seed) and applied in a random order with
    torchvision's functional ops. The mask is left untouched.
    """

    def __init__(self, keys, prob=0.5, brightness=0.2, contrast=0.2,
                 saturation=0.2, hue=0.05, allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def randomize(self, data=None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return
        self._b = self.R.uniform(max(0.0, 1 - self.brightness), 1 + self.brightness)
        self._c = self.R.uniform(max(0.0, 1 - self.contrast), 1 + self.contrast)
        self._s = self.R.uniform(max(0.0, 1 - self.saturation), 1 + self.saturation)
        self._h = self.R.uniform(-self.hue, self.hue)
        self._order = self.R.permutation(4)

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            was_numpy = isinstance(d[key], np.ndarray)
            img = torch.as_tensor(np.asarray(d[key]), dtype=torch.float32)
            for op in self._order:
                if op == 0:
                    img = TF.adjust_brightness(img, self._b)
                elif op == 1:
                    img = TF.adjust_contrast(img, self._c)
                elif op == 2:
                    img = TF.adjust_saturation(img, self._s)
                else:
                    img = TF.adjust_hue(img, self._h)
            img = torch.clamp(img, 0.0, 1.0)
            d[key] = img.numpy() if was_numpy else img
        return d


def _base_steps():
    """Resize + ImageNet normalisation, shared by train and eval pipelines."""
    return [
        Resized(
            keys=["image", "label"],
            spatial_size=(IMG_SIZE, IMG_SIZE),
            mode=("bilinear", "nearest"),
            align_corners=(False, None),
        ),
        NormalizeIntensityd(
            keys="image",
            subtrahend=IMAGENET_MEAN,
            divisor=IMAGENET_STD,
            channel_wise=True,
        ),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]


def build_eval_transforms() -> Compose:
    """Deterministic pipeline for validation and test (no augmentation)."""
    return Compose(_base_steps())


def build_train_transforms(
    seed: int | None = None,
    rotate_deg: float = 15.0,
    flip_prob: float = 0.5,
    rotate_prob: float = 0.5,
    elastic_prob: float = 0.2,
    jitter_prob: float = 0.5,
) -> Compose:
    """Training pipeline: resize, augment, then ImageNet normalisation.

    Augmentations are justified for dermoscopy: lesions have no canonical orientation
    (flips, rotation), dermoscope illumination/colour varies (colour jitter), and soft
    tissue can deform (mild elastic). Masks use nearest-neighbour interpolation throughout
    so they remain binary.
    """
    rot = math.radians(rotate_deg)
    aug = [
        Resized(
            keys=["image", "label"],
            spatial_size=(IMG_SIZE, IMG_SIZE),
            mode=("bilinear", "nearest"),
            align_corners=(False, None),
        ),
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=0),  # vertical
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=1),  # horizontal
        RandRotated(
            keys=["image", "label"],
            range_x=rot,
            prob=rotate_prob,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        Rand2DElasticd(
            keys=["image", "label"],
            prob=elastic_prob,
            spacing=(30, 30),
            magnitude_range=(1, 2),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandColorJitterd(keys=["image"], prob=jitter_prob),
        NormalizeIntensityd(
            keys="image",
            subtrahend=IMAGENET_MEAN,
            divisor=IMAGENET_STD,
            channel_wise=True,
        ),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]
    transform = Compose(aug)
    if seed is not None:
        transform.set_random_state(seed)
    return transform
