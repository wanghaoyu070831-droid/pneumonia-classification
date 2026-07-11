import torch
from torch import nn


class SimpleCNN(nn.Module):
    """
    用于 PneumoniaMNIST 二分类的基础卷积神经网络。
    输入形状：[batch_size, 1, 28, 28]
    输出形状：[batch_size, 2]
    """

    def __init__(self):
        super().__init__()

        # 卷积特征提取部分
        self.features = nn.Sequential(
            # [batch, 1, 28, 28] → [batch, 16, 28, 28]
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            # [batch, 16, 28, 28] → [batch, 16, 14, 14]
            nn.MaxPool2d(kernel_size=2),

            # [batch, 16, 14, 14] → [batch, 32, 14, 14]
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            # [batch, 32, 14, 14] → [batch, 32, 7, 7]
            nn.MaxPool2d(kernel_size=2),
        )

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 32 个通道，每个通道大小为 7×7
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),

            # 随机关闭部分神经元，减轻过拟合
            nn.Dropout(p=0.3),

            # 输出两个分数：正常、肺炎
            nn.Linear(128, 2),
        )

    def forward(self, images):
        features = self.features(images)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    model = SimpleCNN()

    # 模拟一个包含 64 张图片的批次
    sample_images = torch.randn(64, 1, 28, 28)
    outputs = model(sample_images)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(model)
    print("\n输入形状：", sample_images.shape)
    print("输出形状：", outputs.shape)
    print("可训练参数量：", parameter_count)
    print("第一张图片的模型输出：", outputs[0])