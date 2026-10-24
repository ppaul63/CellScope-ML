"""Train the scratch CNN or the pretrained baseline.

Usage:
  python -m ml.train --model pretrained
  python -m ml.train --model scratch
  python -m ml.train --model scratch --overfit-batch     # sanity check: loss should go to ~0
"""
import argparse
import json
import math
import signal
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.common import get_device, load_config, load_norm_stats, resolve, seed_everything
from ml.dataset import CytologyDataset, build_transforms
from ml.models import build_model
from ml.models.pretrained import set_backbone_trainable
from ml.progress import RunTracker

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def run_eval(model, loader, device, criterion):
    model.eval()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item() * len(y))
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": sum(losses) / len(targets),
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }


def _stop_on_sigterm(signum, frame):
    # `kill <pid>` should end a run as cleanly as Ctrl+C, so its status is recorded as stopped
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["scratch", "pretrained"], required=True)
    ap.add_argument("--epochs", type=int, help="override config")
    ap.add_argument("--batch-size", type=int, help="override config")
    ap.add_argument("--overfit-batch", action="store_true", help="train on one batch to check the pipeline")
    args = ap.parse_args()

    cfg = load_config()
    mcfg = cfg["models"][args.model]
    epochs = args.epochs or mcfg["epochs"]
    bs = args.batch_size or mcfg["batch_size"]
    seed_everything(cfg["seed"])
    device = get_device()
    classes = cfg["classes"]
    raw_dir, splits = resolve(cfg["data"]["raw_dir"]), resolve(cfg["data"]["splits_dir"])
    out_dir = resolve(cfg["artifacts_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pretrained backbones expect ImageNet normalisation; the scratch model uses dataset stats.
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if args.model == "pretrained" else load_norm_stats(cfg)
    size = cfg["image_size"]
    train_ds = CytologyDataset(splits / "train.csv", raw_dir, classes, build_transforms(size, mean, std, train=True))
    val_ds = CytologyDataset(splits / "val.csv", raw_dir, classes, build_transforms(size, mean, std, train=False))

    if args.overfit_batch:
        idx = list(range(min(bs, len(train_ds))))
        train_ds = torch.utils.data.Subset(
            CytologyDataset(splits / "train.csv", raw_dir, classes, build_transforms(size, mean, std, train=False)), idx)
        val_ds, epochs = train_ds, args.epochs or 60

    nw = cfg["num_workers"]
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=nw, drop_last=not args.overfit_batch,
                          persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, bs, shuffle=False, num_workers=nw, persistent_workers=nw > 0)

    # Inverse-frequency class weights to counter imbalance (NILM is ~2x ASCUS)
    counts = np.bincount(CytologyDataset(splits / "train.csv", raw_dir, classes, None).targets, minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    smoothing = 0.0 if args.overfit_batch else cfg["label_smoothing"]
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    arch = mcfg["arch"]
    model = build_model(arch, len(classes), pretrained=True).to(device)
    freeze_epochs = 0 if args.overfit_batch else mcfg.get("freeze_epochs", 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    steps_per_epoch = len(train_dl)
    scheduler = cosine_with_warmup(optimizer, mcfg["warmup_epochs"] * steps_per_epoch, epochs * steps_per_epoch)

    tag = f"{args.model}{'_overfit' if args.overfit_batch else ''}"
    writer = SummaryWriter(out_dir / "runs" / f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}")
    ckpt_path = out_dir / f"{tag}_best.pt"
    best_f1, bad_epochs, history, best = -1.0, 0, [], None
    print(f"device={device} arch={arch} train={len(train_ds)} val={len(val_ds)} epochs={epochs} bs={bs}")
    tracker = RunTracker(
        out_dir, cfg.get("model_id", "default"), tag, arch=arch, device=str(device), epochs_total=epochs,
        batches_per_epoch=steps_per_epoch, batch_size=bs, lr=mcfg["lr"], train_size=len(train_ds),
        val_size=len(val_ds), patience=None if args.overfit_batch else mcfg["patience"], checkpoint=ckpt_path.name,
    )
    status, message = "completed", None
    signal.signal(signal.SIGTERM, _stop_on_sigterm)

    try:
        for epoch in range(1, epochs + 1):
            if freeze_epochs:
                set_backbone_trainable(model, epoch > freeze_epochs)
            model.train()
            t0, run_loss, n = time.time(), 0.0, 0
            for i, (x, y) in enumerate(train_dl, start=1):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss = criterion(model(x), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                run_loss += loss.item() * len(y)
                n += len(y)
                tracker.batch(epoch, i)

            val = run_eval(model, val_dl, device, eval_criterion)
            train_loss = run_loss / n
            lr_now = optimizer.param_groups[0]["lr"]
            metrics = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}, "lr": lr_now}
            history.append(metrics)
            writer.add_scalar("loss/train", train_loss, epoch)
            for k, v in val.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_scalar("lr", lr_now, epoch)
            print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} "
                  f"acc {val['acc']:.4f} macroF1 {val['macro_f1']:.4f} | {time.time() - t0:.0f}s")

            stop = False
            if val["macro_f1"] > best_f1:
                best_f1, bad_epochs = val["macro_f1"], 0
                best = {"epoch": epoch, "val_macro_f1": val["macro_f1"], "val_acc": val["acc"]}
                torch.save({
                    "state_dict": model.state_dict(), "arch": arch, "model_kind": args.model,
                    "classes": classes, "image_size": size, "mean": mean, "std": std,
                    "epoch": epoch, "val": val,
                }, ckpt_path)
            else:
                bad_epochs += 1
                if not args.overfit_batch and bad_epochs >= mcfg["patience"]:
                    status = "early_stopped"
                    message = f"No validation macro-F1 improvement for {bad_epochs} epochs."
                    print(f"early stopping: {message}")
                    stop = True
            tracker.epoch(metrics, best, time.time() - t0)
            if stop:
                break
    except KeyboardInterrupt:
        tracker.finish("stopped", "Training was stopped before it finished.")
        raise
    except Exception as e:
        tracker.finish("failed", f"{type(e).__name__}: {e}")
        raise

    (out_dir / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
    print(f"best val macro-F1 {best_f1:.4f} -> {ckpt_path}")
    if args.overfit_batch:
        ok = history[-1]["train_loss"] < 0.05
        message = "Overfit check passed." if ok else "Overfit check failed: loss did not approach 0."
        print("OVERFIT CHECK", "PASSED" if ok else "FAILED (loss did not approach 0)")
    tracker.finish(status, message)


if __name__ == "__main__":
    main()
