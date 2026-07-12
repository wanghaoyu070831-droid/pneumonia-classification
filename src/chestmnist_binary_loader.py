from pathlib import Path

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RANDOM_SEED = 42


class BinaryChestDataset(Dataset):
    """
    将ChestMNIST多标签数据转换为二分类数据：

    0 = 所有疾病标签均为0（No Finding）
    1 = Pneumonia标签为1
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

        label = torch.tensor(
            self.binary_labels[index],
            dtype=torch.long,
        )

        return image, label


def find_pneumonia_index(info):
    for index, label_name in info["label"].items():
        if label_name.lower() == "pneumonia":
            return int(index)

    raise ValueError("未找到Pneumonia标签")


def create_balanced_binary_dataset(
    original_dataset,
    pneumonia_index,
    seed,
):
    labels = original_dataset.labels

    pneumonia_indices = np.where(
        labels[:, pneumonia_index] == 1
    )[0]

    normal_indices = np.where(
        labels.sum(axis=1) == 0
    )[0]

    # 正常样本很多，因此随机抽取与肺炎相同的数量
    rng = np.random.default_rng(seed)

    selected_normal = rng.choice(
        normal_indices,
        size=len(pneumonia_indices),
        replace=False,
    )

    selected_indices = np.concatenate([
        selected_normal,
        pneumonia_indices,
    ])

    binary_labels = np.concatenate([
        np.zeros(
            len(selected_normal),
            dtype=np.int64,
        ),
        np.ones(
            len(pneumonia_indices),
            dtype=np.int64,
        ),
    ])

    # 统一随机打乱
    order = rng.permutation(len(selected_indices))

    selected_indices = selected_indices[order]
    binary_labels = binary_labels[order]

    return BinaryChestDataset(
        original_dataset,
        selected_indices,
        binary_labels,
    )


def get_chestmnist_binary_loaders(batch_size=32):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),
        ),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std,
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std,
        ),
    ])

    info = INFO["chestmnist"]
    DataClass = getattr(
        medmnist,
        info["python_class"],
    )

    # 使用已经下载的64×64 ChestMNIST
    train_original = DataClass(
        split="train",
        root=str(DATA_DIR),
        transform=train_transform,
        download=True,
        size=64,
    )

    val_original = DataClass(
        split="val",
        root=str(DATA_DIR),
        transform=evaluation_transform,
        download=True,
        size=64,
    )

    test_original = DataClass(
        split="test",
        root=str(DATA_DIR),
        transform=evaluation_transform,
        download=True,
        size=64,
    )

    pneumonia_index = find_pneumonia_index(info)

    train_dataset = create_balanced_binary_dataset(
        train_original,
        pneumonia_index,
        seed=RANDOM_SEED,
    )

    val_dataset = create_balanced_binary_dataset(
        val_original,
        pneumonia_index,
        seed=RANDOM_SEED + 1,
    )

    test_dataset = create_balanced_binary_dataset(
        test_original,
        pneumonia_index,
        seed=RANDOM_SEED + 2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = (
        get_chestmnist_binary_loaders()
    )

    images, labels = next(iter(train_loader))

    print("训练集样本数：", len(train_loader.dataset))
    print("验证集样本数：", len(val_loader.dataset))
    print("测试集样本数：", len(test_loader.dataset))

    print("图像批次形状：", images.shape)
    print("标签批次形状：", labels.shape)

    print(
        "当前批次正常数量：",
        (labels == 0).sum().item(),
    )

    print(
        "当前批次肺炎数量：",
        (labels == 1).sum().item(),
    )