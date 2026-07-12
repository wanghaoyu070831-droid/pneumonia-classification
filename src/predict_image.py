from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision import transforms

from resnet_model import create_resnet18
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "resnet18_224_finetuned_best.pt"
)

# 验证集选择出的最终阈值
THRESHOLD = 0.5


def create_transform():
    """
    将外部胸片处理成与PneumoniaMNIST+一致的格式。
    """

    return transforms.Compose([
        # 按短边进行中心裁剪，得到正方形
        transforms.Lambda(
            lambda image: TF.center_crop(
                image,
                min(image.size),
            )
        ),

        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(device):
    """加载训练好的ResNet18模型。"""
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

    return model


def predict_image(image_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"找不到图片：{image_path}"
        )

    # 读取图片并转换成灰度图
    image = Image.open(image_path).convert("L")

    transform = create_transform()

    # [3, 224, 224] → [1, 3, 224, 224]
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    model = load_model(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    normal_probability = probabilities[0].item()
    pneumonia_probability = probabilities[1].item()

    predicted_label = (
        "Pneumonia"
        if pneumonia_probability >= THRESHOLD
        else "Normal"
    )

    print("\n===== Chest X-ray Prediction =====")
    print(f"图片：{image_path}")
    print(f"运行设备：{device}")
    print(f"Normal概率：{normal_probability:.4f}")
    print(f"Pneumonia概率：{pneumonia_probability:.4f}")
    print(f"分类阈值：{THRESHOLD:.3f}")

    print(f"模型预测类别：{predicted_label}")
    print(
        "提示：这是实验模型输出，不是医学诊断。"
    )
    if pneumonia_probability < THRESHOLD:
        print(
        "肺炎得分低于高敏感度筛查阈值。"
    )

    elif pneumonia_probability < 0.5:
        print(
        "肺炎得分超过筛查阈值，但仍低于0.5；"
        "该结果属于不确定区间，不能直接判断为肺炎。"
    )

    else:
        print(
        "肺炎得分超过0.5，但仍只能作为模型实验结果。"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Predict pneumonia from one chest X-ray."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="待预测胸片的JPG或PNG文件路径",
    )

    args = parser.parse_args()

    predict_image(args.image)


if __name__ == "__main__":
    main()