import torch
import torch.nn as nn
import torchvision.transforms as trans
from PIL import Image
import os
from kitti_simba_attack import KittiSimBA, KITTI_TRANSFORM
from VGG import create_vgg_model
from ResNet import create_resnet_model
from MobileNet import create_mobilenet_model

def load_model(model_name):
    """加载训练好的模型"""
    num_classes = 8  # KITTI数据集的类别数
    
    if model_name == 'vgg':
        model = create_vgg_model(num_classes)
        checkpoint = torch.load('best_vgg_model.pth')
    elif model_name == 'mobilenet':
        model = create_mobilenet_model(num_classes)
        checkpoint = torch.load('best_mobilenet_model.pth')
    elif model_name == 'resnet':
        model = create_resnet_model(num_classes)
        checkpoint = torch.load('best_resnet_model.pth')
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    return model

def load_image(image_path):
    """加载并预处理图像"""
    image = Image.open(image_path).convert('RGB')
    image = KITTI_TRANSFORM(image)
    return image.unsqueeze(0)

def save_image(tensor, filename):
    """保存图像"""
    # 反标准化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    
    # 转换为PIL图像并保存
    image = trans.ToPILImage()(tensor.squeeze(0))
    image.save(filename)

def get_class_name(predicted_idx):
    """获取预测类别的名称"""
    classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
    return classes[predicted_idx]

def main():
    # 设置参数
    image_path = "data/汽车1.jpg"  # 替换为您的KITTI图像路径
    model_name = "resnet"  # 可选: "vgg", "mobilenet", "resnet"
    output_dir = "adversarial_examples"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和图像
    model = load_model(model_name)
    image = load_image(image_path)
    
    # 获取原始预测
    with torch.no_grad():
        output = model(image.cuda())
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.cpu()  # 确保predicted在CPU上
        print(f"原始预测类别: {get_class_name(predicted.item())}")
    
    # 创建SimBA攻击器
    attacker = KittiSimBA(model)
    
    # 执行单张图像攻击
    print("开始生成对抗样本...")
    adversarial_image = attacker.simba_single(
        image.squeeze(0),
        predicted,
        num_iters=1000,  # 可以根据需要调整迭代次数
        epsilon=0.3,
        targeted=False
    )
    
    # 保存原始图像和对抗样本
    save_image(image, os.path.join(output_dir, f"original_{model_name}.png"))
    save_image(adversarial_image.unsqueeze(0), os.path.join(output_dir, f"adversarial_{model_name}.png"))
    
    # 验证对抗样本的效果
    with torch.no_grad():
        output = model(adversarial_image.unsqueeze(0).cuda())
        _, predicted = torch.max(output.data, 1)
        print(f"对抗样本预测类别: {get_class_name(predicted.item())}")

if __name__ == "__main__":
    main() 