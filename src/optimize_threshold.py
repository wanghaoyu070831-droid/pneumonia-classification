from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import get_dataloaders
from model import SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "simple_cnn_best.pt"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 医学筛查中，希望肺炎召回率至少达到 95%
TARGET_RECALL = 0.95


def collect_probabilities(model, data_loader, device):
    """收集真实标签和模型预测肺炎的概率。"""
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return np.array(all_labels), np.array(all_probabilities)


def calculate_metrics(labels, probabilities, threshold):
    """使用指定阈值计算分类指标。"""
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    specificity = true_negative / (
        true_negative + false_positive
    )

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "confusion_matrix": matrix.tolist(),
    }

    return metrics


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"分类阈值：{metrics['threshold']:.3f}")
    print(f"准确率：{metrics['accuracy']:.4f}")
    print(f"平衡准确率：{metrics['balanced_accuracy']:.4f}")
    print(f"精确率：{metrics['precision']:.4f}")
    print(f"肺炎召回率：{metrics['recall']:.4f}")
    print(f"正常识别率：{metrics['specificity']:.4f}")
    print(f"F1：{metrics['f1']:.4f}")
    print(f"ROC-AUC：{metrics['roc_auc']:.4f}")
    print("混淆矩阵：", metrics["confusion_matrix"])


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("运行设备：", device)

    _, val_loader, test_loader = get_dataloaders(
        batch_size=64
    )

    model = SimpleCNN().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # 收集验证集和测试集预测概率
    val_labels, val_probabilities = collect_probabilities(
        model,
        val_loader,
        device,
    )

    test_labels, test_probabilities = collect_probabilities(
        model,
        test_loader,
        device,
    )

    # 尝试不同阈值
    candidate_thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    validation_results = [
        calculate_metrics(
            val_labels,
            val_probabilities,
            threshold,
        )
        for threshold in candidate_thresholds
    ]

    # 只保留肺炎召回率达到 95% 的阈值
    eligible_results = [
        result
        for result in validation_results
        if result["recall"] >= TARGET_RECALL
    ]

    # 在满足召回率要求的前提下，选择正常识别率最高者
    best_validation_result = max(
        eligible_results,
        key=lambda result: (
            result["specificity"],
            result["balanced_accuracy"],
        ),
    )

    optimized_threshold = best_validation_result["threshold"]

    # 默认阈值和优化阈值分别在测试集上评价
    default_test_metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        threshold=0.5,
    )

    optimized_test_metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        threshold=optimized_threshold,
    )

    print_metrics(
        "验证集上选出的最佳阈值",
        best_validation_result,
    )

    print_metrics(
        "测试集：默认阈值 0.5",
        default_test_metrics,
    )

    print_metrics(
        "测试集：验证集优化阈值",
        optimized_test_metrics,
    )

    results = {
        "selection_rule": (
            "Maximize validation specificity "
            "subject to pneumonia recall >= 0.95"
        ),
        "validation_best": best_validation_result,
        "test_default": default_test_metrics,
        "test_optimized": optimized_test_metrics,
    }

    output_path = RESULTS_DIR / "threshold_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # 绘制两个测试集混淆矩阵
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    matrices = [
        np.array(default_test_metrics["confusion_matrix"]),
        np.array(optimized_test_metrics["confusion_matrix"]),
    ]

    titles = [
        "Default threshold = 0.500",
        f"Optimized threshold = {optimized_threshold:.3f}",
    ]

    for axis, matrix, title in zip(
        axes,
        matrices,
        titles,
    ):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"],
            ax=axis,
        )

        axis.set_title(title)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")

    plt.tight_layout()

    figure_path = RESULTS_DIR / "threshold_comparison.png"
    plt.savefig(figure_path, dpi=200)
    plt.show()

    print(f"\n数值结果已保存：{output_path}")
    print(f"对比图已保存：{figure_path}")


if __name__ == "__main__":
    main()