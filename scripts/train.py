#!/usr/bin/env python3
"""Обучение классификатора скелета Shift-GCN, опционально multi-task."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exercise_recognition.config import (
    DATA_PROCESSED,
    MODEL_IN_CHANNELS,
    MODELS_DIR,
    NUM_JOINTS,
    SHIFTGCN_WEIGHTS,
    T_FRAMES,
    configure_stdio_utf8,
    exercise_class_names,
)
from exercise_recognition.classifier_loader import (
    build_classifier,
    infer_in_channels_from_state_dict,
)
from exercise_recognition.shift_gcn import ShiftGCNClassifier
from exercise_recognition.manifest_split import stratified_clip_train_val_indices
from exercise_recognition.skeleton_dataset import SkeletonSequenceDataset
from exercise_recognition.training_utils import EmaWeights, focal_nll_loss, forward_logits


def main() -> None:
    configure_stdio_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2.2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.022)
    ap.add_argument("--val-ratio", type=float, default=0.16)
    ap.add_argument("--label-smoothing", type=float, default=0.055)
    ap.add_argument(
        "--focal-gamma",
        type=float,
        default=0.22,
        help="0 — только CE+label_smoothing; по умолчанию лёгкий focal для похожих классов; 1.2–1.5 — сильнее",
    )
    ap.add_argument(
        "--class-weight-power",
        type=float,
        default=1.15,
        help="Вес класса ∝ (1/count)^p; выше p — сильнее акцент на редких классах",
    )
    ap.add_argument("--patience", type=int, default=55, help="early stopping по val_acc")
    ap.add_argument("--min-epochs", type=int, default=45)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-tta-val", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--head-dim", type=int, default=512)
    ap.add_argument("--base-ch", type=int, default=96)
    ap.add_argument("--mid-ch", type=int, default=192)
    ap.add_argument("--out-ch", type=int, default=384)
    ap.add_argument("--no-attn-pool", action="store_true", help="простое mean-пуллирование вместо attention")
    ap.add_argument("--no-extra-block", action="store_true", help="убрать дополнительный блок l11")
    ap.add_argument("--no-temporal-attn", action="store_true", help="без temporal attention после GCN")
    ap.add_argument("--temporal-attn-heads", type=int, default=4)
    ap.add_argument("--mixup", type=float, default=0.36, help="вероятность mixup на батч (0 = выкл)")
    ap.add_argument("--mixup-alpha", type=float, default=0.28, help="Beta(alpha, alpha) для λ")
    ap.add_argument(
        "--cosine-eta-min-mult",
        type=float,
        default=0.05,
        help="после warmup: eta_min = lr × множитель (слишком мало → хвост обучения часто портит val)",
    )
    ap.add_argument("--ema-decay", type=float, default=0.99935, help="0 = без EMA")
    ap.add_argument(
        "--sampler-multiplier",
        type=int,
        default=3,
        help="WeightedRandomSampler: num_samples = len(train)*множитель (больше — дольше эпоха, лучше редкие классы)",
    )
    ap.add_argument("--multi-task", action="store_true", help="слабые метки: фаза + флаг ошибки")
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="путь к .pt (по умолчанию shift_gcn_best.pt)",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto: CUDA при наличии; cpu — только CPU; cuda — ошибка, если GPU нет",
    )
    ap.add_argument(
        "--one-batch",
        action="store_true",
        help="Один шаг train+val и выход (проверка датасета и модели без полного обучения)",
    )
    ap.add_argument(
        "--manifest",
        type=str,
        default="",
        help="путь к manifest.json (по умолчанию data/processed/manifest.json, не manifest_errors)",
    )
    ap.add_argument(
        "--tensorboard",
        type=str,
        default="",
        help="каталог для TensorBoard (пусто = не логировать)",
    )
    ap.add_argument(
        "--wandb-project",
        type=str,
        default="",
        help="проект Weights & Biases (пусто = выкл; pip install wandb, логин)",
    )
    ap.add_argument(
        "--ensemble-with",
        type=str,
        default="",
        help="второй checkpoint (.pt) с тем же списком classes — в training_meta добавится ensemble_weights",
    )
    ap.add_argument(
        "--init-from",
        type=str,
        default="",
        help="путь к .pt (тот же num_class и архитектура) — дообучение вместо с нуля; рекомендуется снизить --lr",
    )
    ap.add_argument(
        "--val-class-metrics",
        action="store_true",
        help="печатать 5 худших классов по recall на валидации каждую эпоху",
    )
    ap.add_argument(
        "--val-f1",
        action="store_true",
        help="считать macro-F1 на валидации (нужен scikit-learn)",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_path = Path(args.output) if args.output else None
    if out_path is None:
        out_path = SHIFTGCN_WEIGHTS
    if args.multi_task and args.mixup > 0:
        print("mixup отключён при --multi-task", flush=True)
        args.mixup = 0.0
    if args.focal_gamma > 0 and args.label_smoothing > 0:
        print(
            "focal loss: label_smoothing не применяется к основной голове классов",
            flush=True,
        )

    manifest = Path(args.manifest) if args.manifest else DATA_PROCESSED / "manifest.json"
    manifest = manifest.resolve()
    if manifest.name == "manifest_errors.json":
        print("Для manifest_errors используйте scripts/train_fault.py (отдельная голова ошибок).")
        sys.exit(1)
    if not manifest.is_file():
        print(f"Нет {manifest} — сначала scripts/extract_skeletons.py")
        sys.exit(1)

    tb_writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_path = Path(args.tensorboard)
        tb_path.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_path))
        print(f"TensorBoard: tensorboard --logdir {tb_path}", flush=True)

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                config={k: getattr(args, k) for k in sorted(vars(args))},
            )
        except Exception as e:
            print(f"wandb отключён: {e}", flush=True)

    if out_path.is_file() and not args.one_batch:
        prev = MODELS_DIR / "classifier_prev.pt"
        try:
            shutil.copy2(out_path, prev)
            print(f"Предыдущие веса сохранены: {prev}", flush=True)
        except OSError as e:
            print(f"Не удалось скопировать старый checkpoint: {e}", flush=True)

    classes = exercise_class_names()
    if not classes:
        print("Пусто data/raw")
        sys.exit(1)

    ds_train = SkeletonSequenceDataset(
        manifest, augment=True, t_frames=T_FRAMES, multi_task=args.multi_task
    )
    ds_val_plain = SkeletonSequenceDataset(
        manifest, augment=False, t_frames=T_FRAMES, multi_task=args.multi_task
    )
    if len(ds_train) < 5:
        print("Слишком мало образцов в manifest")
        sys.exit(1)

    train_indices, val_indices = stratified_clip_train_val_indices(
        ds_train.items, val_ratio=args.val_ratio, seed=42
    )

    train_ds = Subset(ds_train, train_indices)
    val_ds = Subset(ds_val_plain, val_indices)
    print(
        f"train={len(train_ds)}  val={len(val_ds)}  "
        f"(стратификация по классам, val без аугментации)",
        flush=True,
    )

    train_labels = [ds_train.class_to_idx[ds_train.items[i]["label_name"]] for i in train_indices]
    counts = np.bincount(train_labels, minlength=len(classes)).astype(np.float64)
    inv = (1.0 / np.maximum(counts, 1.0)) ** float(args.class_weight_power)
    class_weights = inv * (len(classes) / inv.sum())
    cw_tensor = torch.tensor(class_weights, dtype=torch.float32)
    sample_w = [float(class_weights[y]) for y in train_labels]
    n_draw = max(len(sample_w), len(sample_w) * max(1, args.sampler_multiplier))
    sampler = WeightedRandomSampler(sample_w, num_samples=n_draw, replacement=True)
    print(f"WeightedRandomSampler num_samples={n_draw} (×{args.sampler_multiplier} от train)", flush=True)
    print(
        f"class weights: power={args.class_weight_power} focal_gamma={args.focal_gamma}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("Ошибка: --device cuda, но torch.cuda.is_available()=False", flush=True)
            sys.exit(1)
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Устройство: CPU (обучение будет медленнее)", flush=True)
    model = ShiftGCNClassifier(
        num_class=len(classes),
        num_point=NUM_JOINTS,
        num_person=1,
        in_channels=MODEL_IN_CHANNELS,
        base_ch=args.base_ch,
        mid_ch=args.mid_ch,
        out_ch=args.out_ch,
        dropout=args.dropout,
        head_dim=args.head_dim,
        attn_pool=not args.no_attn_pool,
        extra_block=not args.no_extra_block,
        temporal_attn=not args.no_temporal_attn,
        temporal_attn_heads=args.temporal_attn_heads,
    ).to(device)
    print(
        f"Shift-GCN: C_in={MODEL_IN_CHANNELS} base={args.base_ch} mid={args.mid_ch} out={args.out_ch} "
        f"attn_pool={not args.no_attn_pool} temporal_attn={not args.no_temporal_attn}",
        flush=True,
    )

    init_from = (args.init_from or "").strip()
    seed_best_acc = 0.0
    if init_from:
        ip = Path(init_from).resolve()
        if ip.is_file():
            try:
                ck_init = torch.load(ip, map_location=device, weights_only=False)
            except TypeError:
                ck_init = torch.load(ip, map_location=device)
            sd0 = ck_init.get("model", ck_init)
            cki = ck_init.get("classes")
            if isinstance(cki, list) and [str(x) for x in cki] != [str(x) for x in classes]:
                print(
                    f"init-from: список classes в {ip} не совпадает с data/raw — пропускаю загрузку",
                    flush=True,
                )
            elif isinstance(sd0, dict):
                try:
                    model.load_state_dict(sd0, strict=True)
                    print(f"init-from: загружены веса из {ip}", flush=True)
                except RuntimeError as e:
                    model.load_state_dict(sd0, strict=False)
                    print(
                        f"init-from: частичная загрузка из {ip} (strict=False): {e}",
                        flush=True,
                    )
                raw_va = ck_init.get("val_acc")
                if raw_va is not None:
                    try:
                        seed_best_acc = max(0.0, float(raw_va))
                        print(
                            f"init-from: стартовый порог val_acc={seed_best_acc:.4f} "
                            f"(лучший чекпоинт не перезапишется худшей эпохой)",
                            flush=True,
                        )
                    except (TypeError, ValueError):
                        pass
            else:
                print(f"init-from: нет model в {ip}", flush=True)
        else:
            print(f"init-from: файл не найден: {ip}", flush=True)

    ema: EmaWeights | None = None
    if args.ema_decay > 0.0:
        ema = EmaWeights(model, decay=args.ema_decay)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_ep = min(22, max(3, args.epochs // 6))
    rest = max(1, args.epochs - warmup_ep)
    w_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.2, end_factor=1.0, total_iters=warmup_ep
    )
    c_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=rest, eta_min=args.lr * float(args.cosine_eta_min_mult)
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[w_sched, c_sched], milestones=[warmup_ep]
    )
    cw_dev = cw_tensor.to(device)
    if args.focal_gamma > 0:
        crit = None
    else:
        crit = nn.CrossEntropyLoss(
            weight=cw_dev, label_smoothing=args.label_smoothing
        )
    crit_phase = nn.CrossEntropyLoss()
    crit_err = nn.CrossEntropyLoss()
    w_mt = 0.22

    def cls_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if args.focal_gamma > 0:
            return focal_nll_loss(
                logits, y, class_weights=cw_dev, gamma=args.focal_gamma
            )
        assert crit is not None
        return crit(logits, y)

    best_acc = float(seed_best_acc)
    no_improve = 0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def forward_eval(xb: torch.Tensor, use_ema: bool) -> torch.Tensor:
        backup: dict[str, torch.Tensor] | None = None
        if use_ema and ema is not None:
            backup = ema.apply_to(model)
        try:
            return forward_logits(model, xb, args.no_tta_val)
        finally:
            if backup is not None:
                EmaWeights.restore(model, backup)

    if args.one_batch:
        model.train()
        tb = next(iter(train_loader))
        if args.multi_task:
            x, y, _, _ = tb
            x = x.to(device).unsqueeze(-1)
            out = model(x)
        else:
            x, y = tb
            x = x.to(device).unsqueeze(-1)
            out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        print(
            f"[one-batch train] x={tuple(x.shape)} logits={tuple(logits.shape)}",
            flush=True,
        )
        model.eval()
        with torch.no_grad():
            vb = next(iter(val_loader))
            if args.multi_task:
                x, y, _, _ = vb
                x = x.to(device).unsqueeze(-1)
                out = model(x)
            else:
                x, y = vb
                x = x.to(device).unsqueeze(-1)
                out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            yv = y.to(device)
            pred = logits.argmax(dim=-1)
            acc = (pred == yv).float().mean().item()
        print(
            f"[one-batch val] x={tuple(x.shape)} logits={tuple(logits.shape)} "
            f"batch_acc={acc:.4f}",
            flush=True,
        )
        print("--one-batch: OK", flush=True)
        if tb_writer is not None:
            tb_writer.close()
        if wandb_run is not None:
            wandb_run.finish()
        return

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_tr = 0.0
        n_seen = 0
        for batch in train_loader:
            if args.multi_task:
                x, y, yp, ye = batch
                yp = yp.to(device)
                ye = ye.to(device)
            else:
                x, y = batch
                yp = ye = None
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            x_in = x.unsqueeze(-1)
            if (
                args.mixup > 0.0
                and x.size(0) > 1
                and random.random() < args.mixup
            ):
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                perm = torch.randperm(x.size(0), device=device)
                x_mix = lam * x + (1.0 - lam) * x[perm]
                x_in = x_mix.unsqueeze(-1)
                y_a, y_b = y, y[perm]
                out = model(x_in)
                logits_mix = out[0] if isinstance(out, tuple) else out
                loss = lam * cls_loss(logits_mix, y_a) + (1.0 - lam) * cls_loss(
                    logits_mix, y_b
                )
            else:
                out = model(x_in)
                if args.multi_task and isinstance(out, tuple):
                    logits, lph, ler = out
                    loss = cls_loss(logits, y) + w_mt * crit_phase(lph, yp) + w_mt * crit_err(
                        ler, ye
                    )
                else:
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = cls_loss(logits, y)
            if not torch.isfinite(loss):
                opt.zero_grad()
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(gn):
                opt.zero_grad()
                continue
            opt.step()
            if ema is not None:
                ema.update(model)
            loss_tr += float(loss.detach()) * x.size(0)
            n_seen += x.size(0)
        sched.step()
        loss_tr /= max(1, n_seen)

        model.eval()
        correct = 0
        total = 0
        use_ema_eval = ema is not None
        per_hits = np.zeros(len(classes), dtype=np.int64)
        per_tot = np.zeros(len(classes), dtype=np.int64)
        val_y_list: list[int] = []
        val_p_list: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                if args.multi_task:
                    x, y, _, _ = batch
                else:
                    x, y = batch
                x = x.to(device).unsqueeze(-1)
                y = y.to(device)
                pred = forward_eval(x, use_ema=use_ema_eval).argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
                if args.val_f1:
                    val_y_list.extend(y.view(-1).cpu().numpy().tolist())
                    val_p_list.extend(pred.view(-1).cpu().numpy().tolist())
                if args.val_class_metrics:
                    yv = y.view(-1).cpu().numpy()
                    pv = pred.view(-1).cpu().numpy()
                    for yi, pi in zip(yv, pv):
                        per_tot[int(yi)] += 1
                        if int(pi) == int(yi):
                            per_hits[int(yi)] += 1
        acc = correct / max(1, total)
        val_f1 = None
        if args.val_f1 and total > 0:
            try:
                from sklearn.metrics import f1_score

                val_f1 = float(
                    f1_score(val_y_list, val_p_list, average="macro", zero_division=0)
                )
            except ImportError:
                val_f1 = None
        lr_now = sched.get_last_lr()[0]
        f1_part = f"  val_f1={val_f1:.4f}" if val_f1 is not None else ""
        print(
            f"epoch {epoch:03d}  lr={lr_now:.2e}  train_loss={loss_tr:.4f}  val_acc={acc:.4f}{f1_part}",
            flush=True,
        )
        if args.val_class_metrics and total > 0:
            rec = per_hits.astype(np.float64) / np.maximum(per_tot.astype(np.float64), 1.0)
            order = np.argsort(rec)
            worst = [int(i) for i in order[: min(5, len(classes))]]
            parts = [f"{classes[i]}: recall={rec[i]:.3f} (n={per_tot[i]})" for i in worst]
            print(f"  худшие классы (val): {'; '.join(parts)}", flush=True)
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", loss_tr, epoch)
            tb_writer.add_scalar("val/accuracy", acc, epoch)
            if val_f1 is not None:
                tb_writer.add_scalar("val/f1_macro", val_f1, epoch)
            tb_writer.add_scalar("opt/lr", lr_now, epoch)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": loss_tr,
                    "val_acc": acc,
                    "lr": lr_now,
                }
            )

        improved = acc > best_acc + 1e-6
        if improved:
            best_acc = acc
            no_improve = 0
            sd_save = ema.shadow if ema is not None else model.state_dict()
            payload = {
                "model": {k: v.detach().clone() for k, v in sd_save.items()},
                "arch": model.arch_config(),
                "classes": classes,
                "t_frames": T_FRAMES,
                "in_channels": MODEL_IN_CHANNELS,
                "epoch": epoch,
                "val_acc": acc,
                "ema": ema is not None,
                "model_arch": "shift_gcn",
                "multi_task": args.multi_task,
            }
            torch.save(payload, out_path)
            print(f"  saved {out_path}  (best val_acc={best_acc:.4f})", flush=True)
        else:
            no_improve += 1

        if epoch >= args.min_epochs and no_improve >= args.patience:
            print(
                f"early stopping (no improve {args.patience} epochs), best val_acc={best_acc:.4f}",
                flush=True,
            )
            break

    # Лучшие веса на диске; в памяти модель может быть «хуже» последних эпох.
    if out_path.is_file():
        try:
            ckpt = torch.load(out_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(out_path, map_location=device)
        try:
            eval_model, ic_eval = build_classifier(ckpt, classes, device)
            eval_model.eval()
            if ic_eval != MODEL_IN_CHANNELS:
                print(
                    f"Пропуск матрицы ошибок: в чекпоинте C_in={ic_eval}, "
                    f"а датасет даёт признаки C={MODEL_IN_CHANNELS} "
                    f"(часто из‑за подмены/отката файла весов, например старый 6ch). "
                    f"Проверьте {out_path}.",
                    flush=True,
                )
            else:
                cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
                with torch.no_grad():
                    for batch in val_loader:
                        if args.multi_task:
                            x, y, _, _ = batch
                        else:
                            x, y = batch
                        x = x.to(device).unsqueeze(-1)
                        y = y.to(device)
                        pred = forward_logits(eval_model, x, args.no_tta_val).argmax(
                            dim=-1
                        )
                        for gt, pr in zip(
                            y.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy()
                        ):
                            cm[int(gt), int(pr)] += 1
                total = int(cm.sum())
                diag_acc = float(np.trace(cm) / max(1, total))
                recalls = {}
                for i, name in enumerate(classes):
                    s = int(cm[i].sum())
                    recalls[name] = float(cm[i, i] / s) if s > 0 else 0.0
                pairs: list[tuple[int, str, str]] = []
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        if i != j and cm[i, j] > 0:
                            pairs.append((int(cm[i, j]), classes[i], classes[j]))
                pairs.sort(key=lambda t: t[0], reverse=True)
                conf_path = MODELS_DIR / "val_confusion.json"
                with open(conf_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "classes": classes,
                            "confusion": cm.tolist(),
                            "per_class_recall": recalls,
                            "val_acc_from_cm": diag_acc,
                            "top_mispredictions": [
                                {"count": c, "true": t, "predicted": p}
                                for c, t, p in pairs[:16]
                            ],
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"Матрица ошибок (val, без аугментации): {conf_path}", flush=True)
                print(
                    f"  acc по матрице={diag_acc:.4f} "
                    f"(ожидается близко к лучшему val_acc ~ {best_acc:.4f})",
                    flush=True,
                )
                for c, t, p in pairs[:8]:
                    print(
                        f"  путаница x{c}: истина={t!r} -> предсказано={p!r}",
                        flush=True,
                    )
        except Exception as e:
            print(f"Не удалось построить матрицу ошибок: {e}", flush=True)

    meta = {
        "classes": classes,
        "best_val_acc": best_acc,
        "in_channels": MODEL_IN_CHANNELS,
        "weights": out_path.relative_to(ROOT).as_posix(),
        "val_augment": False,
        "model_arch": "shift_gcn",
        "multi_task": args.multi_task,
        "focal_gamma": args.focal_gamma,
        "class_weight_power": args.class_weight_power,
    }
    if out_path.is_file():
        try:
            ck = torch.load(out_path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(out_path, map_location="cpu")
        if isinstance(ck, dict):
            sd = ck.get("model", ck)
            if isinstance(sd, dict):
                ic_disk = infer_in_channels_from_state_dict(sd)
                if ic_disk is not None:
                    meta["in_channels"] = ic_disk
                    if ic_disk != MODEL_IN_CHANNELS:
                        print(
                            f"ВНИМАНИЕ: в {out_path.name} C_in={ic_disk}, а train.py/датасет "
                            f"рассчитаны на C={MODEL_IN_CHANNELS}. Для обучения с текущими "
                            "признаками переобучите модель или удалите устаревший .pt.",
                            flush=True,
                        )
            arch = ck.get("arch")
            if isinstance(arch, dict):
                meta["classifier_arch"] = arch
            ck_ma = ck.get("model_arch")
            if isinstance(ck_ma, str) and ck_ma.strip():
                meta["model_arch"] = ck_ma.strip()
            if isinstance(ck.get("multi_task"), bool):
                meta["multi_task"] = bool(ck["multi_task"])

    ens = (args.ensemble_with or "").strip()
    if ens:
        p2 = Path(ens).resolve()
        if p2.is_file():
            try:
                ck2 = torch.load(p2, map_location="cpu", weights_only=False)
            except TypeError:
                ck2 = torch.load(p2, map_location="cpu")
            if isinstance(ck2, dict):
                cls2 = ck2.get("classes")
                if isinstance(cls2, list) and [str(x) for x in cls2] == [
                    str(x) for x in classes
                ]:
                    meta["ensemble_weights"] = [p2.relative_to(ROOT).as_posix()]
                    print(
                        f"В training_meta добавлен ensemble_weights: {meta['ensemble_weights']}",
                        flush=True,
                    )
                else:
                    print(
                        "ensemble-with: списки classes не совпадают — ensemble_weights не записан",
                        flush=True,
                    )
            else:
                print("ensemble-with: не удалось прочитать checkpoint", flush=True)
        else:
            print(f"ensemble-with: файл не найден {p2}", flush=True)

    with open(MODELS_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    if out_path.is_file():
        print(
            f"\nЭкспорт ONNX (TensorRT / onnxruntime): "
            f'python scripts/export_classifier_onnx.py --checkpoint "{out_path}" --verify',
            flush=True,
        )


if __name__ == "__main__":
    main()
