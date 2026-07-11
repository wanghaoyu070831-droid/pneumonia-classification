from pathlib import Path

import matplotlib.pyplot as plt
import medmnist
import numpy as np
from medmnist import INFO
from torchvision import transforms


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# PneumoniaMNIST 的基本信息
data_flag = "pneumoniamnist"
info = INFO[data_flag]
DataClass = getattr(medmnist, info["python_class"])

print("任务类型：", info["task"])
print("图像通道数：", info["n_channels"])
print("类别：", info["label"])


# 将图像转换成 PyTorch 张量
transform = transforms.ToTensor()


# 下载并加载三个数据集
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


# 输出数据集大小和类别分布
datasets = {
    "训练集": train_dataset,
    "验证集": val_dataset,
    "测试集": test_dataset,
}

for name, dataset in datasets.items():
    labels = dataset.labels.flatten()
    counts = np.bincount(labels, minlength=2)

    print(f"\n{name}样本数：{len(dataset)}")
    print(f"正常样本：{counts[0]}")
    print(f"肺炎样本：{counts[1]}")


# 展示训练集中的前 8 张图像
fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for index, axis in enumerate(axes.flat):
    image, label = train_dataset[index]
    label_number = int(label.item())
    label_name = info["label"][str(label_number)]

    axis.imshow(image.squeeze(0), cmap="gray")
    axis.set_title(label_name)
    axis.axis("off")

plt.tight_layout()

output_path = RESULTS_DIR / "sample_images.png"
plt.savefig(output_path, dpi=200)
plt.show()

print(f"\n示例图像已保存至：{output_path}")