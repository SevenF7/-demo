import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import numpy as np

# 自定义数据集，读取图像与对应标签（这里取标签文件的第一个单词作为类别）
class KITTIDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        image_dir: 图像文件夹路径
        label_dir: 标签文件夹路径
        transform: 图像预处理变换
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        # 假设图像都是 .png 文件，按照文件名排序
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        # 建立类别映射表
        self.label2idx = {"Car": 0, "Van": 1, "Truck": 2, "Pedestrian": 3, 
                           "Person_sitting": 4, "Cyclist": 5, "Tram": 6, "Misc": 7, "DontCare": 8}

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace('.png', '.txt'))
        
        # 打开图像，并转换为RGB格式
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 读取标签文件，取第一列作为类别
        with open(label_path, 'r') as f:
            content = f.readline().strip().split()
        obj_type = content[0]
        label = self.label2idx.get(obj_type, -1)
        
        return image, label

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 数据目录路径（WSL 中的 Linux 路径）
image_dir = "data/data_object_image_2/training/image_2"
label_dir = "data/training/label_2"

# 创建数据集和 DataLoader
dataset = KITTIDataset(image_dir, label_dir, transform=transform)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# 加载预训练的 ResNet50 模型，并修改最后一层（这里假设输出9个类别）
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 9)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # 使用预训练模型，不进行训练

# 使用 ART 库将模型封装为分类器
classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=None,  # 不需要优化器，因为不训练
    input_shape=(3, 224, 224),
    nb_classes=9,
    clip_values=(0, 1)
)

# 定义 FGSM 攻击
attack_fgsm = FastGradientMethod(estimator=classifier, eps=0.03)

# 从 DataLoader 中获取一批样本
# images, labels = next(iter(dataloader))
# images = images.to(device)

image, label = dataset[2]
image = image.unsqueeze(0)

# 先预测原始样本的输出
with torch.no_grad():
    # outputs_orig = model(images)
    outputs_orig = model(image)
preds_orig = torch.argmax(outputs_orig, dim=1).detach().cpu().numpy()

# 将原始样本转换为 numpy 数组供 ART 使用
# images_np = images.detach().cpu().numpy()
images_np = image.detach().cpu().numpy()

# 生成对抗样本
adv_images = attack_fgsm.generate(x=images_np)

# 对对抗样本进行预测
preds_adv = classifier.predict(adv_images)
preds_adv_idx = np.argmax(preds_adv, axis=1)

print("Original predictions:", preds_orig)
print("Adversarial predictions:", preds_adv_idx)
