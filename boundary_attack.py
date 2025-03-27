import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

def load_model(model_type='vgg', model_path=None, num_classes=8):
    """
    加载预训练模型
    :param model_type: 模型类型 ('vgg', 'resnet' 或 'mobilenet')
    :param model_path: 模型权重文件路径
    :param num_classes: 类别数量
    :return: 加载好的模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'vgg':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_type == 'resnet':
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        # 使用与训练时相同的分类器结构
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif model_type == 'mobilenet':
        model = models.mobilenet_v2(pretrained=False)
        # 修改分类器结构，与训练时保持一致
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        # 直接加载state_dict，不需要修改键名
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    return model

def boundary_attack(model, orig_image, target_image, num_steps=10000, step_size=0.01, spherical_step=0.01, device='cpu', progress_callback=None):
    """
    Boundary Attack 实现。
    :param model: 目标分类模型
    :param orig_image: 原始图像 (未被误分类的样本)
    :param target_image: 目标图像 (被误分类的样本)
    :param num_steps: 迭代次数
    :param step_size: 线性步长
    :param spherical_step: 球面步长
    :param device: 设备 (cpu/gpu)
    :param progress_callback: 进度回调函数
    :return: 生成的对抗样本
    """
    model.eval()
    orig_image = orig_image.to(device)
    target_image = target_image.to(device)
    adversarial = target_image.clone().detach()
    
    # 获取原始图像的预测标签
    with torch.no_grad():
        orig_label = model(orig_image.unsqueeze(0)).argmax(dim=1).item()
        adv_label = model(adversarial.unsqueeze(0)).argmax(dim=1).item()
    
    if orig_label == adv_label:
        raise ValueError("目标图像已经被分类为与原始图像相同的类别！")
    
    for step in range(num_steps):
        # 更新进度
        if progress_callback:
            progress_callback(step, num_steps)
            
        # 计算朝向原始图像的方向
        perturbation = adversarial - orig_image
        perturbation_norm = torch.norm(perturbation)
        if perturbation_norm < 1e-10:
            break
        perturbation = perturbation / perturbation_norm
        
        # 生成正交扰动
        spherical_perturbation = torch.randn_like(adversarial)
        
        # 计算正交投影
        dot_product = torch.sum(spherical_perturbation * perturbation)
        spherical_perturbation -= dot_product * perturbation / (perturbation_norm**2 + 1e-10)
        
        # 归一化
        spherical_perturbation_norm = torch.norm(spherical_perturbation)
        if spherical_perturbation_norm > 1e-10:
            spherical_perturbation = spherical_perturbation * (spherical_step / spherical_perturbation_norm)
        
        # 生成候选样本
        adversarial_candidate = adversarial - step_size * perturbation + spherical_perturbation
        adversarial_candidate = torch.clamp(adversarial_candidate, 0, 1)
        
        # 检查误分类
        with torch.no_grad():
            adv_label_candidate = model(adversarial_candidate.unsqueeze(0)).argmax().item()
        
        if adv_label_candidate != orig_label:
            adversarial = adversarial_candidate
    
    return adversarial.detach()

def preprocess_image(image_path, model_type='vgg'):
    """
    预处理图像
    :param image_path: 图像路径
    :param model_type: 模型类型
    :return: 处理后的图像张量
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def postprocess_image(tensor):
    """
    后处理图像
    :param tensor: 图像张量
    :return: PIL图像
    """
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                          std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
    ])
    return inv_transform(tensor)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 选择要使用的模型
    model_type = 'mobilenet'  # 可选: 'vgg', 'resnet', 'mobilenet'
    model_path = {
        'vgg': "best_vgg_model.pth",
        'resnet': "best_resnet_model.pth",
        'mobilenet': "best_mobilenet_model.pth"
    }[model_type]
    
    # 加载模型
    model = load_model(model_type=model_type, model_path=model_path)
    
    # 加载示例图像
    orig_image = preprocess_image("data/processed_data/train/Car/000003_614_181_727_284.png")
    target_image = preprocess_image("data/processed_data/train/Truck/000019_0_0_424_374.png")
    
    # 生成对抗样本
    adversarial = boundary_attack(model, orig_image, target_image, num_steps=1000, device=device)
    
    # 保存对抗样本
    adv_img = postprocess_image(adversarial)
    adv_img.save(f"adversarial_{model_type}.png")
    
    # 打印预测结果
    with torch.no_grad():
        orig_pred = model(orig_image.unsqueeze(0).to(device)).argmax(dim=1).item()
        adv_pred = model(adversarial.unsqueeze(0).to(device)).argmax(dim=1).item()
        print(f"原始图像预测类别: {orig_pred}")
        print(f"对抗样本预测类别: {adv_pred}")

if __name__ == '__main__':
    main()
