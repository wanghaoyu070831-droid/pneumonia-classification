from pathlib import Path

import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def get_dataloaders(batch_size=64):
    """
    创建训练集、验证集和测试集的 DataLoader。
    """

    # 将图像转换为张量，并归一化到约 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    info = INFO["pneumoniamnist"]
    DataClass = getattr(medmnist, info["python_class"])

    train_dataset = DataClass(
        split="train",
        root=str(DATA_DIR),
        transform=transform,
        download=True,
    )

    val_dataset = DataClass(
        split="val",
        root=str(DATA_DIR),
        transform=transform,
        download=True,
    )

    test_dataset = DataClass(
        split="test",
        root=str(DATA_DIR),
        transform=transform,
        download=True,
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
    train_loader, val_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("一个批次的图像形状：", images.shape)
    print("一个批次的标签形状：", labels.shape)
    print("图像数据类型：", images.dtype)
    print("标签前十项：", labels[:10].flatten())
    print("训练批次数：", len(train_loader))
    print("验证批次数：", len(val_loader))
    print("测试批次数：", len(test_loader))