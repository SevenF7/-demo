import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import foolbox as fb

# 加载预训练模型（以ResNet-18为例）
model = models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

# 加载输入图像（示例：从URL加载）
url = "https://example.com/cat.jpg"  # 替换为实际图片URL
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image = image.resize((224, 224))  # ResNet输入尺寸为224x224
image = np.array(image).astype(np.float32) / 255.0  # 归一化到[0,1]

# 检查模型原始预测
image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
with torch.no_grad():
    logits = model(image_tensor)
original_label = logits.argmax().item()
print(f"Original label: {original_label}")

# 初始化Boundary Attack
attack = fb.attacks.BoundaryAttack(steps=1000)  # 设置迭代次数

# 生成对抗样本
raw, clipped, is_adv = attack(fmodel, image, original_label, epsilons=None)

# 检查对抗样本是否成功
if is_adv:
    adversarial_label = fmodel(clipped).argmax()
    print(f"Adversarial label: {adversarial_label}")
    
    # 可视化结果
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Adversarial Image")
    plt.imshow(clipped)
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Perturbation")
    perturbation = np.clip(10 * (clipped - image), 0, 1)
    plt.imshow(perturbation)
    plt.axis("off")
    
    plt.show()
else:
    print("Attack failed!")