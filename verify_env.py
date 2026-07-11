import sys

import medmnist
import torch
import torchvision


def main() -> None:
    """检查肺炎分类项目所需的基础环境。"""

    print("=" * 50)
    print("环境检查结果")
    print("=" * 50)

    print(f"Python版本：{sys.version}")
    print(f"PyTorch版本：{torch.__version__}")
    print(f"Torchvision版本：{torchvision.__version__}")
    print(
        "MedMNIST版本："
        f"{getattr(medmnist, '__version__', '未提供版本号')}"
    )

    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用：{cuda_available}")

    if cuda_available:
        print(f"GPU型号：{torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("当前使用CPU。基础PneumoniaMNIST实验仍然可以运行。")
        device = torch.device("cpu")

    # 创建一个简单张量，检查PyTorch能否执行计算。
    test_tensor = torch.rand(2, 3).to(device)

    print(f"测试张量：\n{test_tensor}")
    print(f"张量所在设备：{test_tensor.device}")
    print("=" * 50)
    print("环境检查通过。")


if __name__ == "__main__":
    main()