from pathlib import Path
import time

import torch
from torch import nn
from torch.optim import AdamW

from resnet_data_loader import get_resnet_dataloaders
from resnet_model import create_resnet18
from train_baseline import (
    calculate_class_weights,
    evaluate,
    set_random_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device,
):
    """
    训练ResNet18的分类头。

    主干网络保持评价模式，避免预训练BatchNorm参数改变；
    只有最后的分类层处于训练模式。
    """
    model.eval()
    model.fc.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)
        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def main():
    set_random_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("训练设备：", device)

    train_loader, val_loader, _ = (
        get_resnet_dataloaders(batch_size=32)
    )

    model = create_resnet18(
        freeze_backbone=True
    ).to(device)

    class_weights = calculate_class_weights(
        train_loader,
        device,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # 只把最后分类层的参数交给优化器
    optimizer = AdamW(
        model.fc.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    best_val_auc = 0.0
    checkpoint_path = (
        CHECKPOINT_DIR / "resnet18_head_best.pt"
    )

    training_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

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

        epoch_seconds = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"训练损失 {train_loss:.4f} | "
            f"训练准确率 {train_accuracy:.4f} | "
            f"验证损失 {val_metrics['loss']:.4f} | "
            f"验证准确率 {val_metrics['accuracy']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | "
            f"AUC {val_metrics['auc']:.4f} | "
            f"耗时 {epoch_seconds:.1f}秒"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "freeze_backbone": True,
                },
                checkpoint_path,
            )

            print(
                f"  已保存当前最佳模型："
                f"{checkpoint_path}"
            )

    total_seconds = time.perf_counter() - training_start

    print(
        f"\n训练结束，最佳验证集AUC："
        f"{best_val_auc:.4f}"
    )
    print(
        f"总训练时间：{total_seconds / 60:.2f}分钟"
    )


if __name__ == "__main__":
    main()