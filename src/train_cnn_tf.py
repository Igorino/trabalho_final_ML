import os
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# CONFIG
DATA_DIR = "resources/celeba_subset"
OUT_DIR = "results/cnn_torch_baseline"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 5
LR = 1e-3
SEED = 42

GRAYSCALE = False
NUM_WORKERS = 0

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


# Dataset
class CelebASubsetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, label_map: dict,
                 split_name: str, img_size: int = 128, USE_GRAYSCALE: bool = False):
        sub_df = df[df["split"] == split_name].copy()
        if sub_df.empty:
            raise ValueError(f"Nenhuma amostra encontrada para split={split_name}")

        self.paths = sub_df["filepath"].apply(lambda p: os.path.join(data_dir, p)).tolist()
        self.labels = sub_df["person_id"].map(label_map).astype(np.int64).tolist()

        self.img_size = img_size
        self.USE_GRAYSCALE = USE_GRAYSCALE

        print(f"{split_name}: {len(self.paths)} imagens")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Imagem não encontrada: {path}")

        img = Image.open(path)

        if self.USE_GRAYSCALE:
            img = img.convert("L")  # 1 canal
        else:
            img = img.convert("RGB")  # 3 canais

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Para tensor [C,H,W] em float32 normalizado [0,1]
        img_np = np.array(img, dtype=np.float32) / 255.0

        if self.USE_GRAYSCALE:
            # [H,W] -> [1,H,W]
            img_np = np.expand_dims(img_np, axis=0)
        else:
            # [H,W,C] -> [C,H,W]
            img_np = np.transpose(img_np, (2, 0, 1))

        return torch.from_numpy(img_np), torch.tensor(label, dtype=torch.long)


# Model (CNN simples)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Com img 128 -> 64 -> 32 -> 16 (3 pools)
        # canais 128, spatial 16x16 => 128*16*16 = 32768
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Train / Eval
def accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def plot_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    plt.figure()
    plt.plot(df["train_loss"], label="train_loss")
    plt.plot(df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["train_acc"], label="train_acc")
    plt.plot(df["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=150)
    plt.close()


def save_confusion_and_report(y_true, y_pred, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de confusão:")
    print(cm)

    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    df = load_split_csv(DATA_DIR)
    label_map, num_classes = create_label_mapping(df)

    in_channels = 1 if GRAYSCALE else 3

    train_set = CelebASubsetDataset(df, DATA_DIR, label_map, "train", IMG_SIZE, GRAYSCALE)
    val_set   = CelebASubsetDataset(df, DATA_DIR, label_map, "val", IMG_SIZE, GRAYSCALE)
    test_set  = CelebASubsetDataset(df, DATA_DIR, label_map, "test", IMG_SIZE, GRAYSCALE)

    # DataLoaders
    # num_workers=0 é melhor no Windows
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    model = SimpleCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs(OUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(OUT_DIR, "best_model.pt")

    history = []
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # checkpoint + early stopping simples
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"OKAY - Melhorou val_loss. Salvando: {best_path}")
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("PAUSE - Parando antes.")
                break

    plot_history(history, OUT_DIR)

    # carrega melhor modelo
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # avaliação final teste
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    y_true, y_pred = predict_all(model, test_loader, device)
    save_confusion_and_report(y_true, y_pred, OUT_DIR)


if __name__ == "__main__":
    main()
