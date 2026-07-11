from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import Adam

from data_loader import get_dataloaders
from model import SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def set_random_seed(seed):
    """固定随机数，使实验结果更容易复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_class_weights(train_loader, device):
    """
    根据训练集类别数量计算损失权重。

    数量较少的 normal 类别获得更高权重，
    避免模型只倾向预测 pneumonia。
    """
    labels = train_loader.dataset.labels.flatten()
    class_counts = np.bincount(labels)

    total_samples = class_counts.sum()
    weights = total_samples / (2 * class_counts)

    print("训练集类别数量：", class_counts)
    print("损失函数类别权重：", weights)

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    """完成一轮训练。"""
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)

        # 清除上一个批次留下的梯度
        optimizer.zero_grad()

        # 前向传播：计算模型输出
        logits = model(images)

        # 计算预测结果与真实标签之间的损失
        loss = loss_function(logits, labels)

        # 反向传播：计算每个参数的梯度
        loss.backward()

        # 根据梯度更新模型参数
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def evaluate(model, data_loader, loss_function, device):
    """在验证集上评价模型，但不更新参数。"""
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            logits = model(images)
            loss = loss_function(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    average_loss = total_loss / len(data_loader.dataset)

    metrics = {
        "loss": average_loss,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),
        "auc": roc_auc_score(all_labels, all_probabilities),
    }

    return metrics


def main():
    set_random_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("训练设备：", device)

    train_loader, val_loader, _ = get_dataloaders(batch_size=64)

    model = SimpleCNN().to(device)

    class_weights = calculate_class_weights(
        train_loader,
        device,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_auc = 0.0
    checkpoint_path = CHECKPOINT_DIR / "simple_cnn_best.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            loss_function,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"训练损失 {train_loss:.4f} | "
            f"训练准确率 {train_accuracy:.4f} | "
            f"验证损失 {val_metrics['loss']:.4f} | "
            f"验证准确率 {val_metrics['accuracy']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"AUC {val_metrics['auc']:.4f}"
        )

        # 只保存验证集 AUC 最好的模型
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

            print(f"  已保存当前最佳模型：{checkpoint_path}")

    print(f"\n训练结束，最佳验证集 AUC：{best_val_auc:.4f}")


if __name__ == "__main__":
    main()