"""
Preprocessing pipeline for Diabetic Retinopathy classification.

Implements the Ben Graham preprocessing method (Kaggle DR competition winner):
    1. Circle crop  — remove black borders, focus on retina
    2. Resize       — uniform dimensions (default 512x512)
    3. Ben Graham   — subtract local average color to normalize illumination
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import time
import sys


def circle_crop(img):
    """
    Detect the circular retina region and crop out black borders.
    
    Works by:
    1. Convert to grayscale
    2. Threshold to find the bright retinal region
    3. Find the largest contour (the retina)
    4. Fit a minimum enclosing circle
    5. Crop to a tight square around the circle
    
    Returns the cropped image (still rectangular, centered on retina).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)

    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    cx, cy, radius = int(cx), int(cy), int(radius)

    margin = int(radius * 0.02)
    radius += margin

    h, w = img.shape[:2]
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)

    cropped = img[y1:y2, x1:x2]

    # If crop is too small (bad detection), return original
    if cropped.shape[0] < 50 or cropped.shape[1] < 50:
        return img

    return cropped


def ben_graham_preprocess(img, sigmaX=10):
    """
    Ben Graham's preprocessing for retinal images.
    
    Formula: processed = img - GaussianBlur(img, sigmaX) + 128
    
    This subtracts the local average color (illumination component)
    and re-centers around 128, making the image illumination-invariant.
    Vessels and lesions become clearly visible regardless of the 
    original lighting conditions.
    
    Args:
        img: BGR image (uint8)
        sigmaX: Gaussian blur sigma. Higher = more aggressive normalization.
                10 is the standard value from the original implementation.
    
    Returns:
        Preprocessed BGR image (uint8)
    """
    # Gaussian blur to estimate local illumination
    # kernel size 0 means it's auto-computed from sigmaX
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)

    # Subtract local average and re-center at 128
    # Use addWeighted to handle the arithmetic cleanly with clipping
    processed = cv2.addWeighted(
        img, 4,        # source1 * alpha
        blurred, -4,   # source2 * beta  
        128            # gamma (offset)
    )

    return processed


def preprocess_pipeline(img_path, target_size=512, apply_ben_graham=True):
    """
    Full preprocessing pipeline for a single retinal image.
    
    Pipeline:
        1. Load image
        2. Circle crop (remove black borders)
        3. Resize to target_size x target_size
        4. Ben Graham enhancement (normalize illumination)
    
    Args:
        img_path: Path to the input image
        target_size: Output image dimensions (square)
        apply_ben_graham: Whether to apply Ben Graham preprocessing
    
    Returns:
        Preprocessed image (BGR, uint8) or None if loading fails
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = circle_crop(img)

    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    if apply_ben_graham:
        img = ben_graham_preprocess(img, sigmaX=10)

    return img


def batch_preprocess(input_dir, output_dir, target_size=512, apply_ben_graham=True):
    """
    Batch preprocess all images in input_dir and save to output_dir.
    
    Expects input_dir structure:
        input_dir/
            class_1/
                img1.png
                img2.png
            class_2/
                ...
    
    Creates same structure in output_dir with preprocessed images.
    
    Args:
        input_dir: Path to raw image directory
        output_dir: Path to save preprocessed images
        target_size: Output image dimensions (square)
        apply_ben_graham: Whether to apply Ben Graham preprocessing
    
    Returns:
        dict with processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "failed_files": [],
        "per_class": {}
    }

    # Discover all class directories
    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"No class directories found in {input_dir}")
        return stats

    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Ben Graham: {'ON' if apply_ben_graham else 'OFF'}")
    print("-" * 60)

    start_time = time.time()

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        out_cls_dir = output_dir / cls_name
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        images = list(cls_dir.glob("*"))
        stats["per_class"][cls_name] = {"total": len(images), "success": 0, "failed": 0}

        for idx, img_path in enumerate(images):
            stats["total"] += 1

            try:
                processed = preprocess_pipeline(
                    img_path,
                    target_size=target_size,
                    apply_ben_graham=apply_ben_graham
                )

                if processed is None:
                    stats["failed"] += 1
                    stats["failed_files"].append(str(img_path))
                    stats["per_class"][cls_name]["failed"] += 1
                    continue

                # Save as PNG (lossless)
                out_path = out_cls_dir / img_path.name
                cv2.imwrite(str(out_path), processed)

                stats["success"] += 1
                stats["per_class"][cls_name]["success"] += 1

            except Exception as e:
                stats["failed"] += 1
                stats["failed_files"].append(f"{img_path}: {e}")
                stats["per_class"][cls_name]["failed"] += 1

            if (idx + 1) % 100 == 0:
                print(f"  [{cls_name}] {idx + 1}/{len(images)}")

        print(
            f"  {cls_name}: "
            f"{stats['per_class'][cls_name]['success']}/{len(images)} processed"
        )

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"Total: {stats['success']}/{stats['total']} success, {stats['failed']} failed")

    if stats["failed_files"]:
        print(f"\nFailed files:")
        for f in stats["failed_files"][:10]:
            print(f"  - {f}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess retinal images for DR classification"
    )
    parser.add_argument(
        "--input", type=str, default="final_data",
        help="Input directory with class subdirectories (default: final_data)"
    )
    parser.add_argument(
        "--output", type=str, default="processed_data",
        help="Output directory for preprocessed images (default: processed_data)"
    )
    parser.add_argument(
        "--size", type=int, default=512,
        help="Target image size in pixels (default: 512)"
    )
    parser.add_argument(
        "--no-ben-graham", action="store_true",
        help="Skip Ben Graham preprocessing (only crop + resize)"
    )

    args = parser.parse_args()

    batch_preprocess(
        input_dir=args.input,
        output_dir=args.output,
        target_size=args.size,
        apply_ben_graham=not args.no_ben_graham
    )
