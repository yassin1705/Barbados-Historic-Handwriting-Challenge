"""Reproducible augmentation for RGB handwriting line crops.

Severity controls strength, independently of augmentation probability. Zero
returns an exact copy; 0.3 is mild; 1 is the strongest supported setting. It is
not a percentage of changed pixels or a guaranteed perceptual distance.
Regenerate offline variants in a fresh media directory to use this policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import albumentations as A
import cv2
import numpy as np


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 RGB image.")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2:
        raise ValueError("Expected an H x W x 3 RGB image, at least 2 x 2.")


def _uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _paper_color(image: np.ndarray) -> tuple[float, float, float]:
    """Estimate paper from lighter border pixels, avoiding dark ink."""
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    brightness = border.astype(np.float32).mean(axis=1)
    paper = border[brightness >= np.percentile(brightness, 60)]
    return tuple(float(x) for x in np.median(paper, axis=0))


def _geometry(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    operation = str(rng.choice(["rotate", "shear", "stretch", "placement"]))
    amount = float(rng.uniform(0.4, 1.0)) * strength
    direction = float(rng.choice([-1, 1]))
    paper = _paper_color(image)
    if operation == "placement":
        # Add margins instead of shifting text off the canvas.
        margin_x = int(round(height * 0.12 * amount))
        margin_y = int(round(height * 0.12 * amount))
        left = int(rng.integers(margin_x + 1))
        top = int(rng.integers(margin_y + 1))
        return cv2.copyMakeBorder(
            image, top, margin_y - top, left, margin_x - left,
            cv2.BORDER_CONSTANT, value=paper,
        )
    matrix = np.eye(2, 3, dtype=np.float64)
    if operation == "rotate":
        # A fixed angle adds excessive background to very wide strips.
        max_angle = min(1.5, math.degrees(math.atan2(0.18 * height, width)))
        matrix = cv2.getRotationMatrix2D(
            ((width - 1) / 2, (height - 1) / 2), direction * max_angle * amount, 1.0,
        )
    elif operation == "shear":
        matrix[0, 1] = math.tan(math.radians(direction * 6.0 * amount))
    else:
        matrix[0, 0] = 1.0 + direction * 0.12 * amount
    # Fit all pixel centers, with a one-pixel interpolation margin.
    corners = np.array([[0, 0, 1], [width - 1, 0, 1],
                        [0, height - 1, 1], [width - 1, height - 1, 1]])
    transformed = corners @ matrix.T
    lower = np.floor(transformed.min(axis=0)) - 1
    upper = np.ceil(transformed.max(axis=0)) + 1
    matrix[:, 2] -= lower
    output_size = tuple(int(x) for x in upper - lower + 1)
    return cv2.warpAffine(
        image, matrix, output_size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=paper,
    )


def _degradation_tolerance(image: np.ndarray) -> float:
    """Protect small/faint crops; crop height is a proxy for text size."""
    sampled = image[::max(1, image.shape[0] // 100), ::max(1, image.shape[1] // 300)]
    gray = sampled.astype(np.float32).mean(axis=2)
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    return (float(np.clip(image.shape[0] / 120, 0.25, 1.0))
            * float(np.clip(contrast / 50, 0.35, 1.0)))


def _appearance(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    operation = str(rng.choice(["contrast", "gamma", "illumination", "stroke"]))
    amount = float(rng.uniform(0.4, 1.0)) * strength
    direction = float(rng.choice([-1, 1]))
    pixels = image.astype(np.float32)
    if operation == "contrast":
        paper = np.array(_paper_color(image), dtype=np.float32)
        return _uint8(paper + (pixels - paper) * (1 + direction * 0.3 * amount))
    if operation == "gamma":
        return _uint8(255 * (pixels / 255) ** (1 + direction * 0.25 * amount))
    if operation == "illumination":
        # Smooth shading without hard artificial shadow boundaries.
        height, width = image.shape[:2]
        x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
        y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
        mix = float(rng.uniform(0, 1))
        field = 1 + direction * 0.18 * amount * (mix * x + (1 - mix) * y)
        return _uint8(pixels * field[:, :, None])
    # Blend a small stroke change; full erosion can erase superscripts/dots.
    kernel = np.ones((3, 3) if image.shape[0] < 300 else (5, 5), dtype=np.uint8)
    changed = cv2.erode(image, kernel) if direction > 0 else cv2.dilate(image, kernel)
    blend = 0.35 * amount * _degradation_tolerance(image)
    return _uint8(pixels * (1 - blend) + changed.astype(np.float32) * blend)


def _degradation(image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    operation = str(rng.choice(["blur", "noise", "compression"]))
    amount = strength * float(rng.uniform(0.4, 1.0)) * _degradation_tolerance(image)
    if operation == "blur":
        blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0.8)
        # Blend rather than jumping between blur kernels as severity changes.
        return _uint8(image.astype(np.float32) * (1 - amount) + blurred * amount)
    if operation == "noise":
        noise = rng.normal(0, 8 * amount, (*image.shape[:2], 1))
        return _uint8(image.astype(np.float32) + noise)
    quality = int(round(98 - 28 * amount))
    success, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise RuntimeError("Could not encode the compression augmentation.")
    compressed = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return _uint8(image.astype(np.float32) * (1 - amount) + compressed * amount)


class _HandwritingVariation(A.ImageOnlyTransform):
    """One or two groups, using an independent reproducible random stream."""

    def __init__(self, severity: float, two_transform_probability: float, p: float = 1.0):
        super().__init__(p=p)
        self.severity = severity
        self.two_transform_probability = two_transform_probability

    def get_params(self) -> dict:
        return {"sample_seed": int(self.random_generator.integers(0, 2**32))}

    def apply(self, img: np.ndarray, sample_seed: int, **params) -> np.ndarray:
        _validate_image(img)
        result = img.copy()
        if self.severity == 0:
            return result
        rng = np.random.default_rng(sample_seed)
        count = 2 if rng.random() < self.two_transform_probability else 1
        # Without replacement: at most one degradation operation per image.
        groups = rng.choice(3, size=count, replace=False, p=[0.45, 0.40, 0.15])
        operations = (_geometry, _appearance, _degradation)
        seeds = rng.integers(0, 2**32, size=3)
        for group in sorted(groups):
            result = operations[group](result, self.severity, np.random.default_rng(int(seeds[group])))
        return result

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("severity", "two_transform_probability")


@dataclass(frozen=True)
class HistoricalHandwritingAugmentation:
    """Control change strength separately from frequency.

    Defaults: 35% original, 45% one group, 20% two groups, in expectation per call.
    Explicit notebook probabilities override that mix, not severity. Input and
    output are uint8 RGB; geometry may change dimensions. Keep final resizing
    identical between training and inference. The unchanged path of apply()
    returns a copy, although a later JPEG save may itself change pixels.
    """

    severity: float = 0.30
    probability: float = 0.65
    two_transform_probability: float = 4 / 13

    def __post_init__(self) -> None:
        for name in ("severity", "probability", "two_transform_probability"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value!r}.")

    def build(self, *, seed: int | None = None) -> A.Compose:
        # Compose seeds all children identically by default. Separate the outer
        # probability decision and variation streams, even on per-image rebuilds.
        outer, inner = np.random.SeedSequence(seed).spawn(2)
        transform = _HandwritingVariation(self.severity, self.two_transform_probability)
        pipeline = A.Compose(
            [transform], p=self.probability if self.severity > 0 else 0,
            seed=int(outer.generate_state(1)[0]),
        )
        transform.set_random_seed(int(inner.generate_state(1)[0]))
        return pipeline

    def apply(self, image: np.ndarray, *, seed: int) -> np.ndarray:
        """Return a deterministic variant without modifying the input image."""
        _validate_image(image)
        return self.build(seed=seed)(image=image.copy())["image"]
