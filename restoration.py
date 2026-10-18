
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_DIR = BASE_DIR / "dataset" / "original"
RESULTS_DIR = BASE_DIR / "results"

DAMAGED_DIR = RESULTS_DIR / "damaged"
MASK_DIR = RESULTS_DIR / "masks"
RESTORED_DIR = RESULTS_DIR / "restored"
METRICS_DIR = RESULTS_DIR / "metrics"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

for folder in [
    ORIGINAL_DIR, DAMAGED_DIR, MASK_DIR, RESTORED_DIR, METRICS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    return image


def resize_image(image, max_size=512):
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))

    if scale < 1:
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return image.copy()


def create_damage(original, seed=42):
    """Create repeatable synthetic scratches and dust spots."""
    rng = np.random.default_rng(seed)
    damaged = original.copy()
    h, w = damaged.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    # Mild noise
    noise = rng.normal(0, 3.5, damaged.shape).astype(np.float32)
    damaged = np.clip(
        damaged.astype(np.float32) + noise, 0, 255
    ).astype(np.uint8)

    # Mild fading
    damaged = cv2.convertScaleAbs(
        damaged, alpha=0.92, beta=8
    )

    scale = (h * w) / (512 * 512)

    # Scratches
    for _ in range(max(7, int(12 * scale))):
        x1 = int(rng.integers(w))
        y1 = int(rng.integers(h))
        length = int(rng.integers(
            max(10, h // 20), max(11, h // 3)
        ))

        angle = rng.uniform(-np.pi, np.pi)

        x2 = int(np.clip(
            x1 + length * np.cos(angle), 0, w - 1
        ))
        y2 = int(np.clip(
            y1 + length * np.sin(angle), 0, h - 1
        ))

        thickness = int(rng.choice([1, 1, 1, 2]))
        value = int(rng.choice([0, 255]))
        color = (value, value, value)

        cv2.line(
            damaged, (x1, y1), (x2, y2),
            color, thickness, cv2.LINE_AA
        )
        cv2.line(
            mask, (x1, y1), (x2, y2),
            255, thickness, cv2.LINE_AA
        )

    # Dust spots
    for _ in range(max(18, int(55 * scale))):
        x = int(rng.integers(w))
        y = int(rng.integers(h))
        radius = int(rng.choice([1, 1, 2, 2, 3]))
        value = int(rng.choice([0, 255]))
        color = (value, value, value)

        cv2.circle(
            damaged, (x, y), radius,
            color, -1, cv2.LINE_AA
        )
        cv2.circle(
            mask, (x, y), radius,
            255, -1, cv2.LINE_AA
        )

    return damaged, mask


def restore_image(damaged, ground_truth_mask):
    """Restore known synthetic scratches and spots."""
    if damaged is None or ground_truth_mask is None:
        raise ValueError("Damaged image or mask is missing.")

    mask = ground_truth_mask.copy()

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape[:2] != damaged.shape[:2]:
        mask = cv2.resize(
            mask,
            (damaged.shape[1], damaged.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Slight expansion to cover scratch edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Fill masked defects
    restored = cv2.inpaint(
        damaged, mask, 2, cv2.INPAINT_TELEA
    )

    # Gentle denoising to limit detail loss
    restored = cv2.fastNlMeansDenoisingColored(
        restored, None, 2, 2, 7, 15
    )

    return restored, restored.copy()


def calculate_metrics(reference, test):
    """Return MSE, PSNR and SSIM-style comparison score."""
    if reference is None or test is None:
        raise ValueError("Cannot calculate metrics for missing images.")

    if reference.shape != test.shape:
        test = cv2.resize(
            test,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    mse = float(np.mean(
        (reference.astype(np.float64) -
         test.astype(np.float64)) ** 2
    ))

    psnr = (
        float("inf") if mse == 0
        else float(10 * np.log10((255.0 ** 2) / mse))
    )

    x = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY).astype(np.float32)

    ux = cv2.GaussianBlur(x, (11, 11), 1.5)
    uy = cv2.GaussianBlur(y, (11, 11), 1.5)

    vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - ux * ux
    vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - uy * uy
    vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - ux * uy

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    score = (
        (2 * ux * uy + c1) * (2 * vxy + c2)
    ) / (
        (ux * ux + uy * uy + c1) *
        (vx + vy + c2) + 1e-12
    )

    return {
        "MSE": round(mse, 4),
        "PSNR (dB)": round(psnr, 4),
        "SSIM": round(float(np.mean(score)), 4)
    }
