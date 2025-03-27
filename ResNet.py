import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 定义KITTI数据集类
class KITTIDataset(Dataset):
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
        
        # 打印类别分布
        class_counts = {}
        for label in self.labels:
            cls = self.classes[label]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        print("\n类别分布:")
        for cls, count in class_counts.items():
            print(f"{cls}: {count}张图片")

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
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

# 定义ResNet模型
def create_resnet_model(num_classes):
    # 使用预训练的ResNet50模型
    model = models.resnet50(pretrained=True)
    
    # 冻结特征提取层
    for param in model.parameters():
        param.requires_grad = False
        
    # 修改最后一层
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # 使用tqdm显示训练进度
        train_pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # 更新进度条
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        # 使用tqdm显示验证进度
        val_pbar = tqdm(val_loader, desc='Validation')
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 更新进度条
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc)

        # 更新学习率
        scheduler.step(epoch_loss)

        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, 'best_resnet_model.pth')

        print()

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('resnet_training_curves.png')
    plt.close()

def main():
    try:
        # 设置设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 数据预处理
        train_transform, val_transform = get_data_transforms()
     
        # 加载数据集
        base_path = 'data/processed_data'
        train_dataset = KITTIDataset(root_dir=os.path.join(base_path, 'train'), transform=train_transform)
        val_dataset = KITTIDataset(root_dir=os.path.join(base_path, 'val'), transform=val_transform)

        # 使用较小的batch_size
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

        # 创建模型
        num_classes = len(train_dataset.classes)
        model = create_resnet_model(num_classes)
        model = model.to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        # 训练模型
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, device=device)
        
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")

if __name__ == '__main__':
    main()
