from pathlib import Path
import json

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import seaborn as sns
import torch
from medmnist import INFO
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from resnet_model import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "resnet18_224_finetuned_best.pt"
)

THRESHOLD = 0.5
MAX_PER_CLASS = 500
RANDOM_SEED = 42


class ExternalBinaryDataset(Dataset):
    """
    将ChestMNIST多标签测试集转换为：
    0 = No Finding
    1 = Pneumonia
    """

    def __init__(
        self,
        original_dataset,
        selected_indices,
        binary_labels,
    ):
        self.original_dataset = original_dataset
        self.selected_indices = selected_indices
        self.binary_labels = binary_labels

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        original_index = self.selected_indices[index]

        image, _ = self.original_dataset[original_index]
        label = self.binary_labels[index]

        return image, torch.tensor(
            label,
            dtype=torch.long,
        )


def load_external_dataset():
    transform = transforms.Compose([
        # 下载64×64版本，随后调整到模型输入大小
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    info = INFO["chestmnist"]
    DataClass = getattr(
        medmnist,
        info["python_class"],
    )

    test_dataset = DataClass(
        split="test",
        root=str(DATA_DIR),
        transform=transform,
        download=True,
        size=64,
    )

    print("ChestMNIST标签：")
    print(info["label"])

    # 自动寻找Pneumonia对应的标签编号
    pneumonia_index = None

    for index, label_name in info["label"].items():
        if label_name.lower() == "pneumonia":
            pneumonia_index = int(index)
            break

    if pneumonia_index is None:
        raise ValueError("未找到Pneumonia标签")

    labels = test_dataset.labels

    # 肺炎组：Pneumonia标签为1
    pneumonia_indices = np.where(
        labels[:, pneumonia_index] == 1
    )[0]

    # 正常组：所有14种疾病标签都是0
    normal_indices = np.where(
        labels.sum(axis=1) == 0
    )[0]

    print("原始肺炎样本数：", len(pneumonia_indices))
    print("原始No Finding样本数：", len(normal_indices))

    rng = np.random.default_rng(RANDOM_SEED)

    sample_count = min(
        len(pneumonia_indices),
        len(normal_indices),
        MAX_PER_CLASS,
    )

    selected_pneumonia = rng.choice(
        pneumonia_indices,
        size=sample_count,
        replace=False,
    )

    selected_normal = rng.choice(
        normal_indices,
        size=sample_count,
        replace=False,
    )

    selected_indices = np.concatenate([
        selected_normal,
        selected_pneumonia,
    ])

    binary_labels = np.concatenate([
        np.zeros(sample_count, dtype=np.int64),
        np.ones(sample_count, dtype=np.int64),
    ])

    external_dataset = ExternalBinaryDataset(
        original_dataset=test_dataset,
        selected_indices=selected_indices,
        binary_labels=binary_labels,
    )

    return (
        external_dataset,
        selected_indices,
        pneumonia_index,
    )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("外部验证设备：", device)

    (
        external_dataset,
        selected_indices,
        pneumonia_index,
    ) = load_external_dataset()

    data_loader = DataLoader(
        external_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
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

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)

            logits = model(images)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            predictions = (
                probabilities >= THRESHOLD
            ).long()

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp)

    metrics = {
        "dataset": "ChestMNIST test subset",
        "image_size": 64,
        "threshold": THRESHOLD,
        "sample_count": int(len(all_labels)),
        "normal_count": int((all_labels == 0).sum()),
        "pneumonia_count": int((all_labels == 1).sum()),
        "accuracy": float(
            accuracy_score(
                all_labels,
                all_predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                all_labels,
                all_predictions,
            )
        ),
        "precision": float(
            precision_score(
                all_labels,
                all_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                all_labels,
                all_predictions,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                all_labels,
                all_predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                all_labels,
                all_probabilities,
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "pneumonia_label_index": pneumonia_index,
        "selected_indices": selected_indices.tolist(),
    }

    print("\n===== External Validation =====")

    for key in [
        "sample_count",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
    ]:
        value = metrics[key]

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("混淆矩阵：", matrix.tolist())

    result_path = (
        RESULTS_DIR
        / "external_validation_chestmnist.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=["No Finding", "Pneumonia"],
        yticklabels=["No Finding", "Pneumonia"],
    )

    plt.title("External Validation on ChestMNIST")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    figure_path = (
        RESULTS_DIR
        / "external_validation_chestmnist.png"
    )

    plt.savefig(figure_path, dpi=200)
    plt.show()

    print(f"\n结果已保存：{result_path}")
    print(f"图片已保存：{figure_path}")


if __name__ == "__main__":
    main()