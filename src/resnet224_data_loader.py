from pathlib import Path

import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def get_resnet224_dataloaders(batch_size=32):
    """
    加载原生224×224版本的PneumoniaMNIST+。
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        # 数据本身已经是224×224，不再放大
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
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std,
        ),
    ])

    info = INFO["pneumoniamnist"]
    DataClass = getattr(
        medmnist,
        info["python_class"],
    )

    train_dataset = DataClass(
        split="train",
        root=str(DATA_DIR),
        transform=train_transform,
        download=True,
        size=224,
    )

    val_dataset = DataClass(
        split="val",
        root=str(DATA_DIR),
        transform=evaluation_transform,
        download=True,
        size=224,
    )

    test_dataset = DataClass(
        split="test",
        root=str(DATA_DIR),
        transform=evaluation_transform,
        download=True,
        size=224,
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
        get_resnet224_dataloaders()
    )

    images, labels = next(iter(train_loader))

    print(
        "数据集原始数组形状：",
        train_loader.dataset.imgs.shape,
    )
    print("图像批次形状：", images.shape)
    print("标签批次形状：", labels.shape)
    print("图像最小值：", images.min().item())
    print("图像最大值：", images.max().item())
    print("训练样本数：", len(train_loader.dataset))
    print("验证样本数：", len(val_loader.dataset))
    print("测试样本数：", len(test_loader.dataset))