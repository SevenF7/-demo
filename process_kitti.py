import os
import shutil
import random
from PIL import Image
import concurrent.futures
from tqdm import tqdm
import numpy as np

class KITTIProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.image_path = os.path.join(base_path, 'data_object_image_2', 'training', 'image_2')
        self.label_path = os.path.join(base_path, 'training', 'label_2')
        self.output_base = os.path.join(base_path, 'processed_data')
        self.classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        
        # 创建输出目录
        for cls in self.classes:
            os.makedirs(os.path.join(self.output_base, 'train', cls), exist_ok=True)
            os.makedirs(os.path.join(self.output_base, 'val', cls), exist_ok=True)

    def process_single_image(self, img_name):
        """处理单张图片及其标签"""
        # 读取图片
        img_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_path)
        
        # 读取对应的标签文件
        label_name = img_name.replace('.png', '.txt')
        label_path = os.path.join(self.label_path, label_name)
        
        if not os.path.exists(label_path):
            return
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 处理每个标签
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
                
            cls = parts[0]
            if cls not in self.classes:
                continue
                
            # 获取边界框坐标
            left, top, right, bottom = map(float, parts[4:8])
            left, top, right, bottom = map(int, [left, top, right, bottom])
            
            # 确保坐标在图片范围内
            left = max(0, left)
            top = max(0, top)
            right = min(img.width, right)
            bottom = min(img.height, bottom)
            
            if right <= left or bottom <= top:
                continue
                
            # 裁剪图片
            cropped = img.crop((left, top, right, bottom))
            
            # 生成输出文件名
            output_name = f"{img_name[:-4]}_{left}_{top}_{right}_{bottom}.png"
            
            # 随机分配到训练集或验证集
            is_train = random.random() < 0.8
            output_dir = os.path.join(self.output_base, 'train' if is_train else 'val', cls)
            output_path = os.path.join(output_dir, output_name)
            
            # 保存裁剪后的图片
            cropped.save(output_path)

    def process_dataset(self):
        """处理整个数据集"""
        # 获取所有图片文件
        image_files = [f for f in os.listdir(self.image_path) if f.endswith('.png')]
        
        # 使用线程池处理图片
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # 使用tqdm显示进度
            list(tqdm(
                executor.map(self.process_single_image, image_files),
                total=len(image_files),
                desc="Processing images"
            ))

def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 设置基础路径
    base_path = 'data'
    
    # 创建处理器实例
    processor = KITTIProcessor(base_path)
    
    # 处理数据集
    print("开始处理数据集...")
    processor.process_dataset()
    
    # 统计数据集大小
    print("\n数据集统计信息：")
    for split in ['train', 'val']:
        print(f"\n{split}集：")
        total = 0
        for cls in processor.classes:
            count = len(os.listdir(os.path.join(processor.output_base, split, cls)))
            print(f"{cls}: {count}张图片")
            total += count
        print(f"总计: {total}张图片")

if __name__ == '__main__':
    main() 