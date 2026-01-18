import os
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ======================
# CONFIG
# ======================
DATA_DIR = "resources/celeba_subset"
OUT_DIR = "results/resnet18_pretrained"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
PATIENCE = 3
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42

NUM_WORKERS = 0
USE_AMP = True # mixed precision pra acelerar na GPU

FREEZE_BACKBONE_EPOCHS = 2  # treina só a cabeça por 2 epochs, depois libera parte do backbone

COMPUTE_TOP5 = True


# Utils
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_csv(data_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(data_dir, "split.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"split.csv não encontrado em {csv_path}")
    return pd.read_csv(csv_path)


def create_label_mapping(df: pd.DataFrame):
    unique_ids = sorted(df["person_id"].unique())
    label_map = {pid: idx for idx, pid in enumerate(unique_ids)}
    num_classes = len(unique_ids)
    print(f"Número de classes (identidades): {num_classes}")
    return label_map, num_classes


def accuracy_top1(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def accuracy_top5(logits, y):
    # topk retorna indices das k maiores probabilidades
    top5 = torch.topk(logits, k=5, dim=1).indices

    # confere se y aparece entre os 5
    correct = (top5 == y.unsqueeze(1)).any(dim=1).float().mean().item()
    return correct


# Dataset
class CelebASubsetDataset(Dataset):
    def __init__(self, df, data_dir, label_map, split_name, transform):
        sub_df = df[df["split"] == split_name].copy()
        if sub_df.empty:
            raise ValueError(f"Nenhuma amostra para split={split_name}")

        self.paths = sub_df["filepath"].apply(lambda p: os.path.join(data_dir, p)).tolist()
        self.labels = sub_df["person_id"].map(label_map).astype(np.int64).tolist()
        self.transform = transform

        print(f"{split_name}: {len(self.paths)} imagens")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.labels[idx]

        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long)


# Model
def build_resnet18(num_classes: int):
    # weights pretrained do ImageNet
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # troca a última camada (fc) para 2000 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        # congela tudo exceto a head (fc)
        if not name.startswith("fc."):
            param.requires_grad = False


def unfreeze_last_block(model):
    # libera o layer4 (último bloco) + fc
    for name, param in model.named_parameters():
        if name.startswith("layer4.") or name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(model):
    # fine tuning leve
    for param in model.parameters():
        param.requires_grad = True


# Train / Eval
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    loss_sum, acc1_sum, acc5_sum = 0.0, 0.0, 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        loss_sum += loss.item()
        acc1_sum += accuracy_top1(logits, y)
        if COMPUTE_TOP5:
            acc5_sum += accuracy_top5(logits, y)

        n_batches += 1

    out = {
        "loss": loss_sum / max(1, n_batches),
        "acc1": acc1_sum / max(1, n_batches),
    }
    if COMPUTE_TOP5:
        out["acc5"] = acc5_sum / max(1, n_batches)
    return out


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, acc1_sum, acc5_sum = 0.0, 0.0, 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item()
        acc1_sum += accuracy_top1(logits, y)
        if COMPUTE_TOP5:
            acc5_sum += accuracy_top5(logits, y)

        n_batches += 1

    out = {
        "loss": loss_sum / max(1, n_batches),
        "acc1": acc1_sum / max(1, n_batches),
    }
    if COMPUTE_TOP5:
        out["acc5"] = acc5_sum / max(1, n_batches)
    return out


@torch.no_grad()
def predict_all(model, loader, device, max_items=None):
    model.eval()
    y_true, y_pred = [], []
    seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        y_pred.extend(preds)
        y_true.extend(y.numpy().tolist())

        seen += len(preds)
        if max_items is not None and seen >= max_items:
            break

    return np.array(y_true), np.array(y_pred)


def save_plots_and_reports(history, out_dir, y_true, y_pred):
    os.makedirs(out_dir, exist_ok=True)

    # history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    # curvas
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_acc1"], label="train_acc1")
    plt.plot(hist_df["epoch"], hist_df["val_acc1"], label="val_acc1")
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc1_curve.png"), dpi=150)
    plt.close()

    if "train_acc5" in hist_df.columns:
        plt.figure()
        plt.plot(hist_df["epoch"], hist_df["train_acc5"], label="train_acc5")
        plt.plot(hist_df["epoch"], hist_df["val_acc5"], label="val_acc5")
        plt.xlabel("Epoch")
        plt.ylabel("Top-5 Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "acc5_curve.png"), dpi=150)
        plt.close()

    # report (2000 classes => texto enorme, mas ok)
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # .npy -> versão compacta de amostra.
    cm = confusion_matrix(y_true, y_pred)
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

    # versão visual compacta: pega só primeiras 50 classes (pra ter algo pra mostrar)
    k = min(50, cm.shape[0])
    plt.figure(figsize=(8, 6))
    plt.imshow(cm[:k, :k], interpolation="nearest")
    plt.title(f"Confusion Matrix (top-{k} classes slice)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_top{k}.png"), dpi=150)
    plt.close()


# ======================
# Main
# ======================
def main():
    set_seed(SEED)

    torch.backends.cudnn.benchmark = True  # acelera convs com input size fixo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    df = load_split_csv(DATA_DIR)
    label_map, num_classes = create_label_mapping(df)

    # transforms com padrão do ImageNet
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Resize crop pra 224
    # Normalização do ImageNet
    # Augment Simples
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_ds = CelebASubsetDataset(df, DATA_DIR, label_map, "train", train_tf)
    val_ds   = CelebASubsetDataset(df, DATA_DIR, label_map, "val",   eval_tf)
    test_ds  = CelebASubsetDataset(df, DATA_DIR, label_map, "test",  eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )

    model = build_resnet18(num_classes).to(device)

    # treinar só a cabeça
    freeze_backbone(model)

    # Otimizador só nos parâmetros treináveis
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and device.type == "cuda") else None

    os.makedirs(OUT_DIR, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(OUT_DIR, "best_model.pt")
    patience_left = PATIENCE

    history = []

    for epoch in range(1, EPOCHS + 1):
        if epoch == FREEZE_BACKBONE_EPOCHS + 1:
            unfreeze_last_block(model)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR * 0.5,
                weight_decay=WEIGHT_DECAY
            )
            print(" Unfreeze layer4 + fc (fine-tuning leve)")

        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        va = eval_epoch(model, val_loader, criterion, device)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "train_acc1": tr["acc1"],
            "val_acc1": va["acc1"],
            "sec_epoch": dt,
        }
        if COMPUTE_TOP5:
            row["train_acc5"] = tr["acc5"]
            row["val_acc5"] = va["acc5"]

        history.append(row)

        msg = (f"Epoch {epoch}/{EPOCHS} | "
               f"train_loss={tr['loss']:.4f} acc1={tr['acc1']:.4f} | "
               f"val_loss={va['loss']:.4f} acc1={va['acc1']:.4f} | "
               f"time={dt:.1f}s")
        if COMPUTE_TOP5:
            msg += f" | val_acc5={va['acc5']:.4f}"
        print(msg)

        if va["loss"] < best_val - 1e-6:
            best_val = va["loss"]
            torch.save(model.state_dict(), best_path)
            patience_left = PATIENCE
            print(f"OKAY - Melhorou val_loss. Salvando: {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("PAUSE - Parando antes.")
                break

    # Avaliação final
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    te = eval_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {te['loss']:.4f} | Test acc1: {te['acc1']:.4f}")
    if COMPUTE_TOP5:
        print(f"Test acc5: {te['acc5']:.4f}")

    y_true, y_pred = predict_all(model, test_loader, device)
    save_plots_and_reports(history, OUT_DIR, y_true, y_pred)


if __name__ == "__main__":
    main()
