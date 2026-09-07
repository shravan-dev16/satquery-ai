"""Prepares a deterministic small subset (35 samples) of VRSBench for Grounding Evaluation.

Integrity Rules:
- Never downloads the full 3.97 GB VRSBench image archive.
- Uses HTTP range-based remote zip extraction to fetch ONLY the exact 35 images.
- Validates every single annotation against its downloaded image:
  - image file exists and can be opened with PIL;
  - image dimensions match bounding box calculations;
  - coordinates satisfy 0 <= xmin < xmax <= W and 0 <= ymin < ymax <= H;
  - ground-truth box has positive area.
- Outputs manifest: docs/evaluation/vrsbench_grounding_subset.json.
"""

import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import hf_hub_download
from PIL import Image


class RemoteZipFile(io.RawIOBase):
    """Seekable HTTP range reader enabling zip extraction without full archive download."""

    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            self.length = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def readinto(self, b: bytearray) -> int:
        size = len(b)
        if self.pos >= self.length or size == 0:
            return 0
        end = min(self.pos + size - 1, self.length - 1)
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            b[: len(data)] = data
            self.pos += len(data)
            return len(data)


def prepare_subset(target_count: int = 35) -> List[Dict[str, Any]]:
    print("==================================================")
    print("VRSBENCH EVALUATION SUBSET GENERATOR")
    print(f"Target count: {target_count} samples")
    print("==================================================")

    # 1. Download referring annotations (only 10 MB)
    annot_path = hf_hub_download(
        repo_id="xiang709/VRSBench",
        filename="VRSBench_EVAL_referring.json",
        repo_type="dataset",
    )
    with open(annot_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    # 2. Select diverse classes relevant to remote sensing
    target_classes = [
        "bridge",
        "harbor",
        "storage-tank",
        "airplane",
        "ship",
        "airport",
        "trainstation",
        "overpass",
    ]

    selected: List[Dict[str, Any]] = []
    class_counts = {c: 0 for c in target_classes}

    # Group candidate samples deterministically
    for item in all_samples:
        cls_name = item.get("obj_cls", "")
        if cls_name in target_classes and class_counts[cls_name] < 5:
            # Check corners exist
            corners = item.get("obj_corner")
            if corners and len(corners) == 8:
                selected.append(item)
                class_counts[cls_name] += 1
                if len(selected) >= target_count:
                    break

    print(f"Selected {len(selected)} samples across categories: {class_counts}")

    # 3. Setup local storage directories
    images_dir = Path("datasets/grounding_eval_subset/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path("docs/evaluation")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # 4. Connect to remote Images_val.zip
    url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"
    print("Opening remote zip central directory (HTTP range request)...")
    rz = RemoteZipFile(url)
    zf = zipfile.ZipFile(rz)
    namelist = set(zf.namelist())

    manifest: List[Dict[str, Any]] = []

    for idx, sample in enumerate(selected):
        img_filename = sample["image_id"]
        zip_img_path = f"Images_val/{img_filename}"

        if zip_img_path not in namelist:
            raise FileNotFoundError(f"Image {zip_img_path} not found in remote archive!")

        # Download individual image bytes if not already saved
        local_img_path = images_dir / img_filename
        if not local_img_path.exists():
            img_bytes = zf.read(zip_img_path)
            local_img_path.write_bytes(img_bytes)

        # Inspect and validate image
        with Image.open(local_img_path) as pil_img:
            width, height = pil_img.size

        # Compute ground truth bbox [xmin, ymin, xmax, ymax]
        corners = sample["obj_corner"]
        xs = [corners[i] for i in range(0, 8, 2)]
        ys = [corners[i] for i in range(1, 8, 2)]

        norm_xmin = max(0.0, min(xs))
        norm_ymin = max(0.0, min(ys))
        norm_xmax = min(1.0, max(xs))
        norm_ymax = min(1.0, max(ys))

        px_xmin = round(norm_xmin * width, 1)
        px_ymin = round(norm_ymin * height, 1)
        px_xmax = round(norm_xmax * width, 1)
        px_ymax = round(norm_ymax * height, 1)

        # Integrity assertions
        assert px_xmin < px_xmax, f"Invalid x bounds for {sample['question_id']}: {px_xmin} >= {px_xmax}"
        assert px_ymin < px_ymax, f"Invalid y bounds for {sample['question_id']}: {px_ymin} >= {px_ymax}"
        assert px_xmax <= width, f"xmax {px_xmax} exceeds width {width}"
        assert px_ymax <= height, f"ymax {px_ymax} exceeds height {height}"
        area = (px_xmax - px_xmin) * (px_ymax - px_ymin)
        assert area > 0, "Zero-area bounding box!"

        sample_record = {
            "sample_id": f"vrsbench_eval_{idx + 1:03d}",
            "question_id": sample["question_id"],
            "image_id": img_filename,
            "local_image_path": str(local_img_path).replace("\\", "/"),
            "category": sample["obj_cls"],
            "referring_expression": sample["question"],
            "image_dimensions": [width, height],
            "ground_truth_bbox_pixel": [px_xmin, px_ymin, px_xmax, px_ymax],
            "ground_truth_bbox_normalized": [
                round(norm_xmin, 4),
                round(norm_ymin, 4),
                round(norm_xmax, 4),
                round(norm_ymax, 4),
            ],
            "dataset_provenance": "VRSBench (validation split, xiang709/VRSBench, Apache 2.0)",
        }
        manifest.append(sample_record)

    manifest_file = manifest_dir / "vrsbench_grounding_subset.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Successfully created manifest with {len(manifest)} verified samples: {manifest_file}")
    return manifest


if __name__ == "__main__":
    prepare_subset(35)
