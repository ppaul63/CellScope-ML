"""Scan the raw dataset, drop corrupt/duplicate images, and write stratified splits.

Usage:  python -m ml.prepare_data
Expects images under data/raw/<CLASS>/ (any sub-folder depth).
"""
import json
from collections import Counter

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml.common import load_config, resolve

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan(raw_dir, classes):
    rows, corrupt = [], []
    for label in classes:
        files = sorted(p for p in (raw_dir / label).rglob("*") if p.suffix.lower() in IMG_EXTS)
        for path in tqdm(files, desc=f"scan {label}"):
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    phash = imagehash.phash(im)
                    rows.append({"path": str(path.relative_to(raw_dir)), "label": label,
                                 "width": im.width, "height": im.height, "phash": phash})
            except Exception as e:  # unreadable / truncated file
                corrupt.append((str(path), repr(e)))
    return rows, corrupt


def dedupe(rows, max_dist):
    """Drop images whose perceptual hash is within max_dist bits of an earlier image.

    If near-duplicates carry *different* labels, the whole group is dropped (ambiguous ground truth).
    """
    hashes = np.array([int(str(r["phash"]), 16) for r in rows], dtype=np.uint64)
    labels = np.array([r["label"] for r in rows])
    keep = np.ones(len(rows), dtype=bool)
    conflict = np.zeros(len(rows), dtype=bool)
    for i in range(len(rows)):
        if not keep[i]:
            continue
        x = hashes[i + 1:] ^ hashes[i]
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        dup = np.where(dist <= max_dist)[0] + i + 1
        dup = dup[keep[dup]]
        if len(dup):
            keep[dup] = False
            if (labels[dup] != labels[i]).any():
                conflict[i] = True
    n_dup = int((~keep).sum())
    kept = [r for r, k, c in zip(rows, keep, conflict) if k and not c]
    return kept, n_dup, int(conflict.sum())


def compute_norm_stats(raw_dir, paths, size=224, max_images=1000, seed=0):
    rng = np.random.default_rng(seed)
    sample = rng.choice(paths, size=min(max_images, len(paths)), replace=False)
    s, ss, n = np.zeros(3), np.zeros(3), 0
    for p in tqdm(sample, desc="norm stats"):
        with Image.open(raw_dir / p) as im:
            a = np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float64) / 255.0
        s += a.reshape(-1, 3).sum(0)
        ss += (a.reshape(-1, 3) ** 2).sum(0)
        n += a.shape[0] * a.shape[1]
    mean = s / n
    std = np.sqrt(ss / n - mean**2)
    return mean.round(4).tolist(), std.round(4).tolist()


def main():
    cfg = load_config()
    raw_dir, out_dir = resolve(cfg["data"]["raw_dir"]), resolve(cfg["data"]["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = cfg["classes"]

    rows, corrupt = scan(raw_dir, classes)
    if not rows:
        raise SystemExit(f"No images found under {raw_dir}/<CLASS>/ for classes {classes}")
    print(f"found {len(rows)} readable images, {len(corrupt)} corrupt")
    for path, err in corrupt[:20]:
        print("  corrupt:", path, err)

    rows, n_dup, n_conflict = dedupe(rows, cfg["data"]["dedupe_hash_distance"])
    print(f"removed {n_dup} near-duplicates; {n_conflict} cross-label duplicate groups dropped")

    df = pd.DataFrame(rows).drop(columns=["phash", "conflict"], errors="ignore")
    sizes = Counter(zip(df.width, df.height))
    print("most common image sizes:", sizes.most_common(5))

    test_frac, val_frac = cfg["data"]["test_frac"], cfg["data"]["val_frac"]
    train_val, test = train_test_split(df, test_size=test_frac, stratify=df.label, random_state=cfg["seed"])
    train, val = train_test_split(train_val, test_size=val_frac / (1 - test_frac),
                                  stratify=train_val.label, random_state=cfg["seed"])
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split[["path", "label"]].to_csv(out_dir / f"{name}.csv", index=False)

    summary = pd.DataFrame({n: s.label.value_counts() for n, s in [("train", train), ("val", val), ("test", test)]})
    summary.loc["TOTAL"] = summary.sum()
    print(summary.to_string())

    mean, std = compute_norm_stats(raw_dir, train.path.tolist(), cfg["image_size"], seed=cfg["seed"])
    (out_dir / "norm_stats.json").write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    print(f"train mean={mean} std={std}\nsplits written to {out_dir}")


if __name__ == "__main__":
    main()
