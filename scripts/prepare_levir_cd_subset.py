"""Extracts a deterministic 20-pair evaluation subset from LEVIR-CD validation split.

Uses HTTP range requests with buffered readinto (FastRemoteZip) to extract
20 selected image pairs (T1, T2, ground-truth change mask) without downloading
the entire dataset archive.

Generates:
- datasets/levir_cd_eval_subset/A/val_X.png (Time 1)
- datasets/levir_cd_eval_subset/B/val_X.png (Time 2)
- datasets/levir_cd_eval_subset/label/val_X.png (Ground truth mask)
- docs/evaluation/levir_cd_subset.json (Deterministic manifest)
"""

import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
from PIL import Image
import numpy as np


class FastRemoteZip(io.RawIOBase):
    """RawIO wrapper providing seek and buffered read via HTTP Range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            self._size = int(resp.headers["Content-Length"])
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, b: bytearray) -> int:
        size = len(b)
        if self._pos >= self._size:
            return 0
        buf_size = max(size, 512 * 1024)
        end = min(self._pos + buf_size - 1, self._size - 1)
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self._pos}-{end}", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        n = min(size, len(data))
        b[:n] = data[:n]
        self._pos += n
        return n


def prepare_levir_cd_subset(num_samples: int = 20) -> Path:
    base_dir = Path("datasets/levir_cd_eval_subset")
    dir_a = base_dir / "A"
    dir_b = base_dir / "B"
    dir_label = base_dir / "label"

    for d in [dir_a, dir_b, dir_label]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("docs/evaluation/levir_cd_subset.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    url = "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/main/val.zip"
    print(f"Connecting to LEVIR-CD validation split at: {url}", flush=True)
    rz = FastRemoteZip(url)

    selected_samples = []

    with zipfile.ZipFile(rz) as zf:
        # Determine available valid indices
        valid_indices = []
        for i in range(1, 65):
            fname = f"val_{i}.png"
            if f"A/{fname}" in zf.namelist() and f"B/{fname}" in zf.namelist() and f"label/{fname}" in zf.namelist():
                valid_indices.append(i)

        target_indices = valid_indices[:num_samples]
        print(f"Targeting {len(target_indices)} deterministic pairs: val_{target_indices[0]} to val_{target_indices[-1]}", flush=True)

        for idx in target_indices:
            fname = f"val_{idx}.png"
            path_a = dir_a / fname
            path_b = dir_b / fname
            path_l = dir_label / fname

            # Extract T1 (A)
            if not path_a.exists() or path_a.stat().st_size == 0:
                with zf.open(f"A/{fname}") as src:
                    path_a.write_bytes(src.read())

            # Extract T2 (B)
            if not path_b.exists() or path_b.stat().st_size == 0:
                with zf.open(f"B/{fname}") as src:
                    path_b.write_bytes(src.read())

            # Extract Ground Truth Mask (label)
            if not path_l.exists() or path_l.stat().st_size == 0:
                with zf.open(f"label/{fname}") as src:
                    path_l.write_bytes(src.read())

            # Verify integrity
            img_a = Image.open(path_a)
            img_b = Image.open(path_b)
            img_l = Image.open(path_l)

            assert img_a.size == img_b.size == img_l.size, (
                f"Dimension mismatch in sample {fname}: A={img_a.size}, B={img_b.size}, L={img_l.size}"
            )
            w, h = img_a.size

            mask_arr = np.array(img_l)
            changed_pixels = int(np.count_nonzero(mask_arr > 128))
            total_pixels = int(w * h)
            change_ratio_pct = round((changed_pixels / total_pixels) * 100.0, 4)

            record = {
                "sample_id": f"levir_cd_val_{idx:03d}",
                "filename": fname,
                "dataset": "LEVIR-CD",
                "split": "val",
                "license": "CC-BY-4.0",
                "dimensions": [w, h],
                "total_pixels": total_pixels,
                "changed_pixels": changed_pixels,
                "change_ratio_pct": change_ratio_pct,
                "t1_path": str(path_a).replace("\\", "/"),
                "t2_path": str(path_b).replace("\\", "/"),
                "gt_mask_path": str(path_l).replace("\\", "/"),
            }
            selected_samples.append(record)
            print(f"[{len(selected_samples)}/{len(target_indices)}] Sample {record['sample_id']}: {w}x{h}, changed: {changed_pixels} px ({change_ratio_pct}%)", flush=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(selected_samples, f, indent=2)

    print(f"\nSuccessfully prepared LEVIR-CD evaluation subset ({len(selected_samples)} samples) in: {manifest_path}", flush=True)
    return manifest_path


if __name__ == "__main__":
    prepare_levir_cd_subset(num_samples=20)
