"""Calibrate raw PRNU residual stats on labeled real/fake folders."""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

from .prnu_sensor_noise import extract_noise_residual, local_noise_variance_map


def load_images(folder):
    imgs = []
    if not os.path.isdir(folder):
        return imgs
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            imgs.append((fname, img))
    return imgs


def extract_features(image):
    residual = extract_noise_residual(image)
    noise_map = local_noise_variance_map(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return {
        "residual_std": float(residual.std()),
        "residual_mean_abs": float(np.abs(residual).mean()),
        "residual_entropy": float(
            -np.sum(np.abs(residual) * np.log2(np.abs(residual) + 1e-10))
        ),
        "noise_var_mean": float(noise_map.mean()),
        "noise_var_std": float(noise_map.std()),
        "noise_var_max": float(noise_map.max()),
        "hf_energy": float(np.sum(np.abs(cv2.Laplacian(gray, cv2.CV_64F)))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Folder with real/ and fake/")
    args = parser.parse_args(argv)
    real_imgs = load_images(os.path.join(args.dataset, "real"))
    fake_imgs = load_images(os.path.join(args.dataset, "fake"))
    print(f"Real images: {len(real_imgs)}")
    print(f"Fake images: {len(fake_imgs)}")
    if not real_imgs or not fake_imgs:
        print("Need both real/ and fake/ image folders.")
        return 1
    real_feats, fake_feats = [], []
    for fname, img in real_imgs:
        f = extract_features(img)
        f["filename"], f["label"] = fname, "real"
        real_feats.append(f)
    for fname, img in fake_imgs:
        f = extract_features(img)
        f["filename"], f["label"] = fname, "fake"
        fake_feats.append(f)
    out_json = os.path.join(args.dataset, "features.json")
    with open(out_json, "w") as fh:
        json.dump(real_feats + fake_feats, fh, indent=2)
    keys = [
        "residual_std", "residual_mean_abs", "residual_entropy",
        "noise_var_mean", "noise_var_std", "noise_var_max", "hf_energy",
    ]
    print(f"{'Feature':<22} {'Real mean':>12} {'Fake mean':>12} {'d':>8}")
    separation = {}
    for key in keys:
        rv = [f[key] for f in real_feats]
        fv = [f[key] for f in fake_feats]
        r_mean, r_std = np.mean(rv), np.std(rv)
        f_mean, f_std = np.mean(fv), np.std(fv)
        pooled = np.sqrt((r_std**2 + f_std**2) / 2) if (r_std + f_std) > 0 else 1e-6
        d = abs(r_mean - f_mean) / pooled
        separation[key] = d
        print(f"{key:<22} {r_mean:>12.4f} {f_mean:>12.4f} {d:>8.3f}")
    best = max(separation, key=separation.get)
    print(f"Most discriminative: {best} d={separation[best]:.3f}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
