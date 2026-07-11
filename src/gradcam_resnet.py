from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from resnet_data_loader import get_resnet_dataloaders
from resnet_model import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "resnet18_finetuned_best.pt"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.165


class GradCAM:
    """生成ResNet18最后一层的Grad-CAM热力图。"""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.hook = target_layer.register_forward_hook(
            self.save_activations
        )

    def save_activations(self, module, inputs, output):
        self.activations = output
        output.register_hook(self.save_gradients)

    def save_gradients(self, gradients):
        self.gradients = gradients

    def generate(self, image, target_class):
        # 让输入参与梯度计算
        image = image.clone().detach()
        image.requires_grad_(True)

        self.model.zero_grad()

        logits = self.model(image)
        target_score = logits[:, target_class].sum()

        target_score.backward()

        # 每个特征通道的平均梯度作为权重
        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * self.activations
        ).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        # 放大到原图尺寸
        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()

        # 归一化到[0,1]
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def close(self):
        self.hook.remove()


def restore_image(normalized_image):
    """将ImageNet归一化后的图像恢复到可显示范围。"""
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = normalized_image.cpu() * std + mean
    image = image.clamp(0, 1)

    # 三通道原本来自同一张灰度图
    return image.mean(dim=0).numpy()


def find_representative_cases(model, dataset, device):
    """
    寻找TN、FP、FN、TP各一个样本。
    """
    cases = {}

    model.eval()

    for index in range(len(dataset)):
        image, label = dataset[index]
        true_label = int(label.item())

        with torch.no_grad():
            logits = model(
                image.unsqueeze(0).to(device)
            )

            probability = torch.softmax(
                logits,
                dim=1,
            )[0, 1].item()

        predicted_label = int(
            probability >= THRESHOLD
        )

        if true_label == 0 and predicted_label == 0:
            case_name = "True Negative"
        elif true_label == 0 and predicted_label == 1:
            case_name = "False Positive"
        elif true_label == 1 and predicted_label == 0:
            case_name = "False Negative"
        else:
            case_name = "True Positive"

        if case_name not in cases:
            cases[case_name] = {
                "image": image,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "probability": probability,
            }

        if len(cases) == 4:
            break

    return cases


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("运行设备：", device)

    _, _, test_loader = get_resnet_dataloaders(
        batch_size=32
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

    cases = find_representative_cases(
        model,
        test_loader.dataset,
        device,
    )

    # 使用ResNet18最后一个残差块
    gradcam = GradCAM(
        model,
        model.layer4[-1],
    )

    case_order = [
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive",
    ]

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(15, 7),
    )

    for column, case_name in enumerate(case_order):
        case = cases[case_name]

        image = case["image"]
        predicted_label = case["predicted_label"]
        probability = case["probability"]

        original_image = restore_image(image)

        cam = gradcam.generate(
            image.unsqueeze(0).to(device),
            target_class=1,)       
        axes[0, column].imshow(
            original_image,
            cmap="gray",
        )
        axes[0, column].set_title(case_name)
        axes[0, column].axis("off")

        axes[1, column].imshow(
            original_image,
            cmap="gray",
        )
        axes[1, column].imshow(
            cam,
            cmap="jet",
            alpha=0.45,
        )

        axes[1, column].set_title(
            f"P(pneumonia)={probability:.3f}"
)
        axes[1, column].axis("off")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR / "resnet18_gradcam.png"
    )

    plt.savefig(output_path, dpi=200)
    plt.show()

    gradcam.close()

    print(f"Grad-CAM已保存：{output_path}")


if __name__ == "__main__":
    main()