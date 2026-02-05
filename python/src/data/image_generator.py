from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: Optional[int] = None


def discover_images(root: Path, extensions: Sequence[str] = IMAGE_EXTENSIONS) -> List[Path]:
    """
    Recursively discover image files under a root directory.
    """
    root = Path(root)
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(root.rglob(f"*{ext}"))
    return sorted(paths)


def default_image_loader(path: Path):
    """
    Default image loader using Pillow. Imported lazily to avoid hard dependency at import time.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to load images.") from exc

    return Image.open(path).convert("RGB")


def build_image_transform(image_size: int, train: bool = False) -> Callable:
    """
    Build a simple torchvision transform pipeline.
    """
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("torchvision is required to build image transforms.") from exc

    ops: List[Callable] = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
        ])
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def build_records_from_folders(
    root: Path,
    class_names: Optional[Sequence[str]] = None,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> List[ImageRecord]:
    """
    Build ImageRecord list from a folder structure: root/class_name/*.jpg
    """
    root = Path(root)
    if class_names is None:
        class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    records: List[ImageRecord] = []

    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for path in discover_images(class_dir, extensions=extensions):
            records.append(ImageRecord(path=path, label=class_to_idx[class_name]))

    return records


def minmax_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Min-max normalize an array to [0, 1]. If range is ~0, return 0.5 for all.
    """
    values = np.asarray(values, dtype=np.float32)
    min_val = float(values.min())
    max_val = float(values.max())
    if (max_val - min_val) < eps:
        return np.full_like(values, 0.5, dtype=np.float32)
    return (values - min_val) / (max_val - min_val)


def _require_pillow():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required to generate images.") from exc
    return Image, ImageDraw


def _to_pixel(value: float, size: int) -> int:
    """
    Convert normalized [0, 1] value to pixel coordinate (0 at top).
    """
    value = max(0.0, min(1.0, value))
    return int(round((1.0 - value) * (size - 1)))


def generate_candlestick_image(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    image_size: int = 224,
) -> "Image.Image":
    """
    Generate a grayscale candlestick chart image from OHLC data.
    """
    Image, ImageDraw = _require_pillow()

    open_prices = np.asarray(open_prices, dtype=np.float32)
    high_prices = np.asarray(high_prices, dtype=np.float32)
    low_prices = np.asarray(low_prices, dtype=np.float32)
    close_prices = np.asarray(close_prices, dtype=np.float32)

    min_price = float(low_prices.min())
    max_price = float(high_prices.max())
    price_range = max_price - min_price
    if price_range < 1e-8:
        price_range = 1.0

    def normalize_price(price: float) -> float:
        return (price - min_price) / price_range

    width = height = image_size
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)

    num_points = len(open_prices)
    margin = max(1, int(width * 0.05))
    available = max(1, width - 2 * margin)
    step = available / max(1, num_points)
    body_width = max(1, int(step * 0.6))

    for i in range(num_points):
        center_x = margin + (i + 0.5) * step
        x = int(round(center_x))

        y_high = _to_pixel(normalize_price(high_prices[i]), height)
        y_low = _to_pixel(normalize_price(low_prices[i]), height)
        draw.line([(x, y_high), (x, y_low)], fill=0)

        y_open = _to_pixel(normalize_price(open_prices[i]), height)
        y_close = _to_pixel(normalize_price(close_prices[i]), height)
        top = min(y_open, y_close)
        bottom = max(y_open, y_close)
        if top == bottom:
            bottom = min(height - 1, top + 1)

        left = int(round(center_x - body_width / 2))
        right = int(round(center_x + body_width / 2))
        draw.rectangle([left, top, right, bottom], fill=0)

    return image


def generate_gaf_image(close_prices: np.ndarray, image_size: int = 224) -> "Image.Image":
    """
    Generate a Gramian Angular Field (GAF) grayscale image from close prices.
    """
    Image, _ = _require_pillow()

    close_prices = np.asarray(close_prices, dtype=np.float32)
    normalized = minmax_normalize(close_prices)
    scaled = np.clip(2.0 * normalized - 1.0, -1.0, 1.0)

    phi = np.arccos(scaled)
    gaf = np.cos(phi[:, None] + phi[None, :])

    gaf_img = ((gaf + 1.0) / 2.0 * 255.0).astype(np.uint8)
    image = Image.fromarray(gaf_img, mode="L")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    return image


def generate_rp_image(close_prices: np.ndarray, image_size: int = 224) -> "Image.Image":
    """
    Generate a recurrence plot (RP) grayscale image from close prices.
    """
    Image, _ = _require_pillow()

    close_prices = np.asarray(close_prices, dtype=np.float32)
    normalized = minmax_normalize(close_prices)
    distances = np.abs(normalized[:, None] - normalized[None, :])

    threshold = float(np.percentile(distances, 10))
    rp = (distances <= threshold).astype(np.uint8) * 255

    image = Image.fromarray(rp, mode="L")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample=Image.NEAREST)
    return image
