"""Evaluate trained checkpoints on the held-out test set and pick the winner.

Usage:  python -m ml.evaluate            # evaluates every artifacts/{scratch,pretrained}_best.pt found
        python -m ml.evaluate --no-tta
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from ml.common import get_device, load_config, resolve
from ml.dataset import CytologyDataset, build_transforms
from ml.gradcam import gradcam, overlay
from ml.models import load_checkpoint

NORMAL = "NILM"
HIGH_GRADE = ("HSIL", "SCC")


@torch.no_grad()
def predict(model, loader, device, tta):
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).softmax(1)
        if tta:  # average over flips; cells are orientation-invariant
            p = p + model(x.flip(3)).softmax(1) + model(x.flip(2)).softmax(1)
            p = p / 3
        probs.append(p.cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def compute_metrics(probs, targets, classes):
    preds = probs.argmax(1)
    normal = classes.index(NORMAL)
    high = [classes.index(c) for c in HIGH_GRADE]
    abnormal_true, abnormal_pred = targets != normal, preds != normal
    report = classification_report(targets, preds, target_names=classes, output_dict=True, zero_division=0)
    return {
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(targets, probs, multi_class="ovr")) if len(set(targets)) == len(classes) else None,
        # Screening view: was an abnormal slide flagged as abnormal at all?
        "abnormal_sensitivity": float((abnormal_pred & abnormal_true).sum() / max(1, abnormal_true.sum())),
        "abnormal_specificity": float((~abnormal_pred & ~abnormal_true).sum() / max(1, (~abnormal_true).sum())),
        "high_grade_recall": float(np.isin(preds[np.isin(targets, high)], high).mean()) if np.isin(targets, high).any() else None,
        "per_class": {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")} for c in classes},
        "confusion_matrix": confusion_matrix(targets, preds, labels=range(len(classes))).tolist(),
    }


def plot_confusion(cm, classes, path, title):
    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_gradcam_samples(model, ckpt, ds, device, path, per_class=2):
    tf = build_transforms(ckpt["image_size"], ckpt["mean"], ckpt["std"], train=False)
    classes = ckpt["classes"]
    picks = [i for c in range(len(classes)) for i in [j for j, t in enumerate(ds.targets) if t == c][:per_class]]
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 2, figsize=(5, 2.5 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, i in enumerate(picks):
        with Image.open(ds.raw_dir / ds.df.path.iloc[i]) as im:
            img = im.convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        cam = gradcam(model, x)
        pred = classes[int(model(x).argmax(1))]
        shown = img.resize((ckpt["image_size"], ckpt["image_size"]))
        axes[row, 0].imshow(shown)
        axes[row, 0].set_title(f"true {classes[ds.targets[i]]}", fontsize=8)
        axes[row, 1].imshow(overlay(shown, cam))
        axes[row, 1].set_title(f"pred {pred}", fontsize=8)
        for a in axes[row]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    device = get_device()
    out_dir = resolve(cfg["artifacts_dir"])
    raw_dir, splits = resolve(cfg["data"]["raw_dir"]), resolve(cfg["data"]["splits_dir"])
    results = {}

    for kind in ("pretrained", "scratch"):
        path = out_dir / f"{kind}_best.pt"
        if not path.exists():
            continue
        model, ckpt = load_checkpoint(path, device)
        classes = ckpt["classes"]
        ds = CytologyDataset(splits / "test.csv", raw_dir, classes,
                             build_transforms(ckpt["image_size"], ckpt["mean"], ckpt["std"], train=False))
        dl = DataLoader(ds, 64, shuffle=False, num_workers=cfg["num_workers"])
        probs, targets = predict(model, dl, device, tta=not args.no_tta)
        m = compute_metrics(probs, targets, classes)
        m.update({"arch": ckpt["arch"], "checkpoint": path.name, "tta": not args.no_tta, "test_size": len(ds)})
        results[kind] = m
        plot_confusion(m["confusion_matrix"], classes, out_dir / f"{kind}_confusion.png", f"{kind} ({ckpt['arch']})")
        save_gradcam_samples(model, ckpt, ds, device, out_dir / f"{kind}_gradcam.png")
        print(f"\n== {kind} ({ckpt['arch']}) ==")
        print(json.dumps({k: v for k, v in m.items() if k not in ("per_class", "confusion_matrix")}, indent=2))

    if not results:
        raise SystemExit(f"No checkpoints found in {out_dir}. Train first: python -m ml.train --model pretrained")

    winner = max(results, key=lambda k: results[k]["macro_f1"])
    (out_dir / "metrics.json").write_text(json.dumps({"winner": winner, "models": results}, indent=2))

    lines = ["# Test-set evaluation", "", f"Winner (by macro-F1): **{winner}**", "",
             "| Metric | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for key in ("accuracy", "macro_f1", "roc_auc_ovr", "abnormal_sensitivity", "abnormal_specificity", "high_grade_recall"):
        vals = [results[k][key] for k in results]
        lines.append(f"| {key} | " + " | ".join("-" if v is None else f"{v:.4f}" for v in vals) + " |")
    for kind, m in results.items():
        lines += ["", f"## {kind} per-class", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for c, r in m["per_class"].items():
            lines.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
        lines += ["", f"![confusion]({kind}_confusion.png)", f"![gradcam]({kind}_gradcam.png)"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\nwinner: {winner}; report -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
