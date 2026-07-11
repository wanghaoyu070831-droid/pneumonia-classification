import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


def create_resnet18(freeze_backbone=True):
    """
    创建用于肺炎二分类的 ResNet18。

    freeze_backbone=True：
    冻结预训练特征提取部分，只训练最后的分类层。
    """

    # 加载在 ImageNet 上预训练的参数
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # ResNet18 原分类层输入特征数为512
    input_features = model.fc.in_features

    # 将原来的1000分类层替换成二分类层
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(input_features, 2),
    )

    return model


if __name__ == "__main__":
    model = create_resnet18(
        freeze_backbone=True
    )

    # 模拟两张三通道224×224图像
    sample_images = torch.randn(
        2,
        3,
        224,
        224,
    )

    model.eval()

    with torch.no_grad():
        outputs = model(sample_images)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("新的分类层：")
    print(model.fc)

    print("\n输入形状：", sample_images.shape)
    print("输出形状：", outputs.shape)
    print("模型总参数量：", total_parameters)
    print("当前可训练参数量：", trainable_parameters)