from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from optimize_threshold import (
    calculate_metrics,
    collect_probabilities,
    print_metrics,
)
from resnet_data_loader import get_resnet_dataloaders
from resnet224_data_loader import get_resnet224_dataloaders
from resnet_model import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "resnet18_224_finetuned_best.pt"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_RECALL = 0.95


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("测试设备：", device)

    _, val_loader, test_loader = (
        get_resnet224_dataloaders(batch_size=32)
)


    model = create_resnet18(
        freeze_backbone=True
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "加载的微调模型来自第几轮：",
        checkpoint["epoch"],
    )
    print(
        "对应验证集指标：",
        checkpoint["val_metrics"],
    )

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

    # 在验证集上搜索分类阈值
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

    eligible_results = [
        result
        for result in validation_results
        if result["recall"] >= TARGET_RECALL
    ]

    best_validation_result = max(
        eligible_results,
        key=lambda result: (
            result["specificity"],
            result["balanced_accuracy"],
        ),
    )

    optimized_threshold = (
        best_validation_result["threshold"]
    )

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
        "ResNet18验证集最佳阈值",
        best_validation_result,
    )

    print_metrics(
        "ResNet18测试集：默认阈值",
        default_test_metrics,
    )

    print_metrics(
        "ResNet18测试集：优化阈值",
        optimized_test_metrics,
    )

    results = {
        "model": "Fine-tuned ResNet18 on native 224x224 images",
        "fine_tuned_layer": "layer4",
        "selection_rule": (
            "Maximize validation specificity "
            "subject to pneumonia recall >= 0.95"
        ),
        "validation_best": best_validation_result,
        "test_default": default_test_metrics,
        "test_optimized": optimized_test_metrics,
    }

    result_path = (
        RESULTS_DIR / "resnet18_224_results.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    matrices = [
        np.array(
            default_test_metrics["confusion_matrix"]
        ),
        np.array(
            optimized_test_metrics["confusion_matrix"]
        ),
    ]

    titles = [
        "ResNet18: threshold = 0.500",
        (
            "ResNet18: threshold = "
            f"{optimized_threshold:.3f}"
        ),
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    for axis, matrix, title in zip(
        axes,
        matrices,
        titles,
    ):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"],
            ax=axis,
        )

        axis.set_title(title)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")

    plt.tight_layout()

    figure_path = (
        RESULTS_DIR
        / "resnet18_224_threshold_comparison.png"
    )

    plt.savefig(figure_path, dpi=200)
    plt.show()

    print(f"\n结果已保存：{result_path}")
    print(f"图片已保存：{figure_path}")


if __name__ == "__main__":
    main()