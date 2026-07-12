from pathlib import Path
import time

import torch
from torch import nn
from torch.optim import AdamW

from resnet_data_loader import get_resnet_dataloaders
from resnet224_data_loader import get_resnet224_dataloaders
from resnet_model import create_resnet18
from train_baseline import (
    calculate_class_weights,
    evaluate,
    set_random_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

HEAD_CHECKPOINT = (
    CHECKPOINT_DIR / "resnet18_224_head_best.pt"
)

OUTPUT_CHECKPOINT = (
    CHECKPOINT_DIR / "resnet18_224_finetuned_best.pt"
)

EPOCHS = 5
RANDOM_SEED = 42


def unfreeze_layer4(model):
    """
    解冻ResNet18最后一个残差阶段和分类层。
    """
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device,
):
    """
    只训练layer4和分类层。
    更早的网络层保持冻结和评价模式。
    """
    model.eval()
    model.layer4.train()
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
    print("微调设备：", device)

    train_loader, val_loader, _ = (
        get_resnet224_dataloaders(batch_size=32)
    )

    # 先创建相同结构
    model = create_resnet18(
        freeze_backbone=True
    ).to(device)

    # 加载已训练好的分类头
    checkpoint = torch.load(
        HEAD_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "加载分类头模型，来自第",
        checkpoint["epoch"],
        "轮",
    )

    # 解冻最后一个残差阶段
    unfreeze_layer4(model)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("微调阶段可训练参数量：", trainable_parameters)

    class_weights = calculate_class_weights(
        train_loader,
        device,
    )

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # layer4使用较小学习率，避免破坏预训练参数
    # 新分类层可以使用稍大的学习率
    optimizer = AdamW(
        [
            {
                "params": model.layer4.parameters(),
                "lr": 0.00001,
            },
            {
                "params": model.fc.parameters(),
                "lr": 0.0001,
            },
        ],
        weight_decay=0.0001,
    )

    best_val_auc = 0.0
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
                    "fine_tuned_layer": "layer4",
                },
                OUTPUT_CHECKPOINT,
            )

            print(
                f"  已保存最佳微调模型："
                f"{OUTPUT_CHECKPOINT}"
            )

    total_seconds = time.perf_counter() - training_start

    print(
        f"\n微调结束，最佳验证集AUC："
        f"{best_val_auc:.4f}"
    )

    print(
        f"总微调时间："
        f"{total_seconds / 60:.2f}分钟"
    )


if __name__ == "__main__":
    main()