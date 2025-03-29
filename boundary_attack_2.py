import torch
import torch.nn as nn
import torchvision
import foolbox as fb
import numpy as np
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义KITTI数据集类
class KITTIDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 获取所有图像文件
        self.image_files = []
        self.labels = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if os.path.exists(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith('.png'):
                        self.image_files.append(os.path.join(cls_dir, img_name))
                        self.labels.append(self.class_to_idx[cls])
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {root_dir} 中没有找到任何图片文件！")
        
        print(f"加载了 {len(self.image_files)} 个图片文件")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据预处理
def get_data_transforms():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def preprocess_image(image_path):
    """
    预处理单张图片
    :param image_path: 图片路径
    :return: 处理后的图片张量
    """
    transform = get_data_transforms()
    image = Image.open(image_path).convert('RGB')
    return transform(image)

# 加载模型
def load_model(model_name, num_classes):
    if model_name == 'vgg':
        model = torchvision.models.vgg16(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        checkpoint = torch.load('best_vgg_model.pth')
    elif model_name == 'resnet':
        model = torchvision.models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        checkpoint = torch.load('best_resnet_model.pth')
    elif model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        checkpoint = torch.load('best_mobilenet_model.pth')
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def perform_boundary_attack(model_name, num_samples=100):
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据集
    transform = get_data_transforms()
    dataset = KITTIDataset(root_dir='data/processed_data/val', transform=transform)
    
    # 加载模型
    model = load_model(model_name, num_classes=len(dataset.classes))
    model = model.to(device)

    # 创建foolbox模型，注意bounds的设置
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))  # 改回原始范围

    # 创建攻击，调整参数
    attack = fb.attacks.BoundaryAttack(
        steps=25000,  # 减少步数，加快测试
        spherical_step=1e-1,  # 增大步长
        source_step=1e-1,  # 增大步长
        source_step_convergance=1e-5,  # 调整收敛条件
        step_adaptation=1.5,
        tensorboard=False,
        update_stats_every_k=5
    )

    # 创建结果目录
    results_dir = f'boundary_attack_results/{model_name}'
    os.makedirs(results_dir, exist_ok=True)

    # 记录攻击统计信息
    total_samples = 0
    successful_attacks = 0
    failed_attacks = 0

    # 对每个样本进行攻击
    for i in tqdm(range(min(num_samples, len(dataset))), desc=f"对{model_name}进行Boundary Attack"):
        # 获取样本
        image, label = dataset[i]
        image = image.unsqueeze(0).to(device)
        # 将标签转换为tensor并移动到正确的设备
        label = torch.tensor([label], dtype=torch.long).to(device)
        
        # 确保模型预测正确
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1).item()
            # 获取原始预测的概率分布
            original_probs = torch.softmax(output, dim=1)
            original_top3 = torch.topk(original_probs, 3)
            
        if pred != label.item():
            print(f"样本 {i} 的原始预测错误，跳过")
            continue

        total_samples += 1

        # 进行攻击
        try:
            # 设置epsilons参数，使用更大的值
            epsilons = [1.0]  # 增大扰动范围
            adversarial = attack(fmodel, image, label, epsilons=epsilons)
            
            # 验证攻击是否成功
            with torch.no_grad():
                adv_output = model(adversarial)
                adv_pred = adv_output.argmax(dim=1).item()
                # 获取对抗样本的预测概率分布
                adv_probs = torch.softmax(adv_output, dim=1)
                adv_top3 = torch.topk(adv_probs, 3)
            
            # 打印详细的分类信息
            print(f"\n样本 {i} 的分类信息:")
            print(f"真实标签: {dataset.classes[label.item()]}")
            print("\n原始预测:")
            for j in range(3):
                print(f"  {dataset.classes[original_top3.indices[0][j]]}: {original_top3.values[0][j]:.4f}")
            
            print("\n对抗样本预测:")
            for j in range(3):
                print(f"  {dataset.classes[adv_top3.indices[0][j]]}: {adv_top3.values[0][j]:.4f}")
            
            if adv_pred != label.item():
                successful_attacks += 1
                print(f"\n攻击成功！原始类别: {dataset.classes[label.item()]} -> 对抗类别: {dataset.classes[adv_pred]}")
                
                # 保存原始图像和对抗样本
                original_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                adversarial_np = adversarial.squeeze().cpu().numpy().transpose(1, 2, 0)
                
                # 反归一化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                original_np = original_np * std + mean
                adversarial_np = adversarial_np * std + mean
                
                # 裁剪到[0,1]范围
                original_np = np.clip(original_np, 0, 1)
                adversarial_np = np.clip(adversarial_np, 0, 1)
                
                # 保存图像
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(original_np)
                plt.title(f'原始图像\n{dataset.classes[label.item()]} ({original_top3.values[0][0]:.4f})')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(adversarial_np)
                plt.title(f'对抗样本\n{dataset.classes[adv_pred]} ({adv_top3.values[0][0]:.4f})')
                plt.axis('off')
                plt.savefig(os.path.join(results_dir, f'sample_{i}.png'))
                plt.close()
                
                # 计算L2距离
                l2_distance = np.sqrt(np.sum((original_np - adversarial_np) ** 2))
                print(f"L2距离: {l2_distance:.4f}")
            else:
                failed_attacks += 1
                print(f"\n攻击未成功，模型仍然预测为原始类别: {dataset.classes[label.item()]}")
            
        except Exception as e:
            failed_attacks += 1
            print(f"样本 {i} 攻击失败: {str(e)}")
            continue

    # 打印统计信息
    print(f"\n攻击统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"成功攻击数: {successful_attacks}")
    print(f"失败攻击数: {failed_attacks}")
    if total_samples > 0:
        success_rate = (successful_attacks / total_samples) * 100
        print(f"攻击成功率: {success_rate:.2f}%")

def perform_targeted_boundary_attack(model_name, source_image_path, target_class, num_steps=50000):
    """
    进行目标类别的 Boundary Attack
    :param model_name: 模型名称 ('vgg', 'resnet', 'mobilenet')
    :param source_image_path: 源图片路径
    :param target_class: 目标类别名称
    :param num_steps: 攻击步数
    """
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据集以获取类别信息
    transform = get_data_transforms()
    dataset = KITTIDataset(root_dir='data/processed_data/val', transform=transform)

    # 检查目标类别是否有效
    if target_class not in dataset.classes:
        raise ValueError(f"无效的目标类别: {target_class}。有效类别为: {dataset.classes}")
    
    target_idx = dataset.class_to_idx[target_class]
    
    # 加载模型
    model = load_model(model_name, num_classes=len(dataset.classes)).to(device)
    model.eval()

    # 创建 foolbox 模型
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # 创建 Boundary Attack（修改参数）
    attack = fb.attacks.BoundaryAttack(
        steps=num_steps,
        spherical_step=1e-1,  # 增大球形步长
        source_step=1e-1,    # 增大源步长
        source_step_convergance=1e-7,  # 放宽收敛条件
        step_adaptation=1.5,  # 调整步长适应率
        tensorboard=False,
        update_stats_every_k=10
    )

    # 预处理源图片
    source_image = preprocess_image(source_image_path).unsqueeze(0).to(device)

    # 反标准化，将图像转换回[0,1]范围
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    source_image = source_image * std + mean
    source_image = torch.clamp(source_image, 0, 1)

    # 获取原始预测
    with torch.no_grad():
        # 重新标准化图像用于模型预测
        normalized_image = (source_image - mean) / std
        output = model(normalized_image)
        orig_pred = output.argmax(dim=1).item()
        orig_probs = torch.softmax(output, dim=1)
        orig_top3 = torch.topk(orig_probs, 3)
    
    print(f"\n原始图片分类信息: 预测类别: {dataset.classes[orig_pred]}")
    for j in range(3):
        print(f"  {dataset.classes[orig_top3.indices[0][j]]}: {orig_top3.values[0][j]:.4f}")

    # 进行攻击（尝试不同的 epsilons）
    epsilons = [0.2, 0.5, 1.0, 2.0, 3.0]  # 增大扰动范围
    adversarial_list, _, success = attack(fmodel, source_image, torch.tensor([target_idx], dtype=torch.long).to(device), epsilons=epsilons)
    
    # 检查攻击是否成功
    if not success.any().item():
        print("攻击未成功，未能生成有效的对抗样本")
        return None

    # 选择第一个成功的对抗样本
    print(len(adversarial_list))
    # print(adversarial_list)
    adversarial = adversarial_list[1]

    # 验证攻击结果
    with torch.no_grad():
        # 标准化对抗样本用于模型预测
        normalized_adversarial = (adversarial - mean) / std
        adv_output = model(normalized_adversarial)
        adv_pred = adv_output.argmax(dim=1).item()
        adv_probs = torch.softmax(adv_output, dim=1)
        adv_top3 = torch.topk(adv_probs, 3)
    
    print(f"\n对抗样本分类信息: 预测类别: {dataset.classes[adv_pred]}")
    for j in range(3):
        print(f"  {dataset.classes[adv_top3.indices[0][j]]}: {adv_top3.values[0][j]:.4f}")
    
    # 计算 L2 距离
    l2_distance = torch.norm((source_image - adversarial).view(-1), p=2).item()
    print(f"\nL2 距离: {l2_distance:.4f}")

    # 保存结果
    results_dir = f'boundary_attack_results/{model_name}'
    os.makedirs(results_dir, exist_ok=True)
    
    # 转换图像格式并保存
    original_np = source_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    adversarial_np = adversarial.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # 保存图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_np)
    plt.title(f'原始图像\n{dataset.classes[orig_pred]} ({orig_top3.values[0][0]:.4f})')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_np)
    plt.title(f'对抗样本\n{dataset.classes[adv_pred]} ({adv_top3.values[0][0]:.4f})')
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, f'targeted_attack_{target_class}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return adversarial


if __name__ == '__main__':
    # 示例使用
    model_name = 'vgg'  # 可选: 'vgg', 'resnet', 'mobilenet'
    source_image_path = 'data/汽车1.jpg'  # 替换为您的源图片路径
    target_class = 'Truck'  # 替换为您想要的目标类别
    
    print(f"\n开始对{model_name}进行目标Boundary Attack...")
    print(f"源图片: {source_image_path}")
    print(f"目标类别: {target_class}")
    
    perform_targeted_boundary_attack(model_name, source_image_path, target_class, num_steps=20000)
