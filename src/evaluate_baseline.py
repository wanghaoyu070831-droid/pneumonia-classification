from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data_loader import get_dataloaders
from model import SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "simple_cnn_best.pt"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("测试设备：", device)

    _, _, test_loader = get_dataloaders(batch_size=64)

    model = SimpleCNN().to(device)

    # 该模型文件由我们自己训练生成，因此可以安全加载
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("加载的最佳模型来自第几轮：", checkpoint["epoch"])
    print("对应验证集指标：", checkpoint["val_metrics"])

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            logits = model(images)

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    metrics = {
        "accuracy": float(
            accuracy_score(all_labels, all_predictions)
        ),
        "precision": float(
            precision_score(all_labels, all_predictions)
        ),
        "recall": float(
            recall_score(all_labels, all_predictions)
        ),
        "f1": float(
            f1_score(all_labels, all_predictions)
        ),
        "roc_auc": float(
            roc_auc_score(all_labels, all_probabilities)
        ),
    }

    print("\n测试集最终指标")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n分类报告")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["normal", "pneumonia"],
            digits=4,
        )
    )

    # 保存数值指标
    metrics_path = RESULTS_DIR / "baseline_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # 计算混淆矩阵
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    # 计算 ROC 曲线
    false_positive_rate, true_positive_rate, _ = roc_curve(
        all_labels,
        all_probabilities,
    )

    # 绘制混淆矩阵和 ROC 曲线
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Pneumonia"],
        yticklabels=["Normal", "Pneumonia"],
        ax=axes[0],
    )

    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    axes[1].plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"Simple CNN (AUC = {metrics['roc_auc']:.4f})",
    )

    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier",
    )

    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    figure_path = RESULTS_DIR / "baseline_evaluation.png"
    plt.savefig(figure_path, dpi=200)
    plt.show()

    print(f"\n指标已保存：{metrics_path}")
    print(f"评价图已保存：{figure_path}")


if __name__ == "__main__":
    main()