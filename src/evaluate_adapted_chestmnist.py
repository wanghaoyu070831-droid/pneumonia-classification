from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from chestmnist_binary_loader import (
    get_chestmnist_binary_loaders,
)
from optimize_threshold import (
    calculate_metrics,
    collect_probabilities,
    print_metrics,
)
from resnet_model import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "resnet18_chestmnist_adapted_best.pt"
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("评价设备：", device)

    _, val_loader, test_loader = (
        get_chestmnist_binary_loaders(
            batch_size=32
        )
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
    model.eval()

    print(
        "加载的最佳域适配模型来自第几轮：",
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

    # 只在验证集上搜索阈值
    candidate_thresholds = np.linspace(
        0.01,
        0.99,
        197,
    )

    validation_results = [
        calculate_metrics(
            val_labels,
            val_probabilities,
            threshold,
        )
        for threshold in candidate_thresholds
    ]

    # 平衡数据任务：最大化平衡准确率
    best_validation_result = max(
        validation_results,
        key=lambda result: (
            result["balanced_accuracy"],
            result["f1"],
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
        "验证集最佳平衡阈值",
        best_validation_result,
    )

    print_metrics(
        "测试集：默认阈值0.5",
        default_test_metrics,
    )

    print_metrics(
        "测试集：验证集优化阈值",
        optimized_test_metrics,
    )

    results = {
        "model": "ChestMNIST-adapted ResNet18",
        "source_model": (
            "PneumoniaMNIST+ native 224 ResNet18"
        ),
        "adaptation_data": "ChestMNIST 64",
        "checkpoint_epoch": checkpoint["epoch"],
        "selection_rule": (
            "Maximize balanced accuracy "
            "on the validation split"
        ),
        "validation_best": best_validation_result,
        "test_default": default_test_metrics,
        "test_optimized": optimized_test_metrics,
    }

    result_path = (
        RESULTS_DIR
        / "chestmnist_adapted_results.json"
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
        "Adapted model: threshold = 0.500",
        (
            "Adapted model: threshold = "
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
            cmap="Greens",
            xticklabels=[
                "No Finding",
                "Pneumonia",
            ],
            yticklabels=[
                "No Finding",
                "Pneumonia",
            ],
            ax=axis,
        )

        axis.set_title(title)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")

    plt.tight_layout()

    figure_path = (
        RESULTS_DIR
        / "chestmnist_adapted_comparison.png"
    )

    plt.savefig(
        figure_path,
        dpi=200,
    )
    plt.show()

    print(f"\n结果已保存：{result_path}")
    print(f"图片已保存：{figure_path}")


if __name__ == "__main__":
    main()