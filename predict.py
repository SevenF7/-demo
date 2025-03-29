import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from boundary_attack import boundary_attack, preprocess_image, postprocess_image
import torch.nn.functional as F

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class KITTIPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path):
        """加载预训练模型"""
        # 根据模型文件名判断模型类型
        if 'vgg' in model_path.lower():
            model = models.vgg16(pretrained=False)
            # 修改分类器
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, len(self.classes))
            )
        elif 'resnet' in model_path.lower():
            model = models.resnet50(pretrained=False)
            # 修改最后一层
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, len(self.classes))
            )
        elif 'mobilenet' in model_path.lower():
            model = models.mobilenet_v2(pretrained=False)
            # 修改分类器
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(1280, 512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, len(self.classes))
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_path}")
        
        # 加载保存的模型状态
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        return model
    
    def predict(self, image_path):
        """对单张图片进行预测"""
        # 加载和预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # 获取预测结果和概率
        predicted_class = self.classes[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()
        
        # 获取所有类别的概率
        all_probabilities = probabilities[0].cpu().numpy()
        
        return predicted_class, confidence, all_probabilities

class PredictorGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("KITTI数据集图像分类预测与对抗样本生成")
        self.window.geometry("1800x1000")  # 增加窗口大小
        
        # 初始化预测器为None
        self.predictor = None
        
        # 创建GUI元素
        self.create_widgets()
        
    def create_widgets(self):
        # 创建主框架
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 创建左右分栏
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # 创建预测部分
        self.create_prediction_section(left_frame)
        
        # 创建对抗样本生成部分
        self.create_adversarial_section(right_frame)
        
    def create_prediction_section(self, parent):
        # 创建滚动框架
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 预测部分标题
        tk.Label(scrollable_frame, text="图像分类预测", font=('Arial', 14, 'bold')).pack(pady=10)
        
        # 模型选择下拉框
        model_frame = tk.Frame(scrollable_frame)
        model_frame.pack(pady=10)
        
        tk.Label(model_frame, text="选择模型:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var)
        self.model_combo['values'] = ['VGG16', 'ResNet50', 'MobileNetV2']
        self.model_combo.pack(side=tk.LEFT, padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select)
        
        # 选择图片按钮
        self.select_button = tk.Button(
            scrollable_frame,
            text="选择图片",
            command=self.select_image,
            width=20,
            height=2
        )
        self.select_button.pack(pady=20)
        
        # 显示图片路径
        self.path_label = tk.Label(scrollable_frame, text="未选择图片", wraplength=400)
        self.path_label.pack(pady=10)
        
        # 图片预览区域
        self.image_label = tk.Label(scrollable_frame)
        self.image_label.pack(pady=10)
        
        # 预测按钮
        self.predict_button = tk.Button(
            scrollable_frame,
            text="开始预测",
            command=self.predict_image,
            width=20,
            height=2
        )
        self.predict_button.pack(pady=20)
        
        # 显示预测结果
        self.result_text = tk.Text(scrollable_frame, height=10, width=50)
        self.result_text.pack(pady=20)
        
        # 创建概率条形图
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=scrollable_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=20)
        
        # 配置滚动区域
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_adversarial_section(self, parent):
        # 创建滚动框架
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 对抗样本生成部分标题
        tk.Label(scrollable_frame, text="对抗样本生成", font=('Arial', 14, 'bold')).pack(pady=10)
        
        # 攻击方法选择
        attack_frame = tk.Frame(scrollable_frame)
        attack_frame.pack(pady=10)
        
        tk.Label(attack_frame, text="攻击方法:").pack(side=tk.LEFT)
        self.attack_var = tk.StringVar(value="Boundary Attack")
        self.attack_combo = ttk.Combobox(attack_frame, textvariable=self.attack_var, state='readonly')
        self.attack_combo['values'] = ['Boundary Attack', 'SimBA']
        self.attack_combo.pack(side=tk.LEFT, padx=5)
        
        # 目标类别选择
        target_frame = tk.Frame(scrollable_frame)
        target_frame.pack(pady=10)
        
        tk.Label(target_frame, text="目标类别:").pack(side=tk.LEFT)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(target_frame, textvariable=self.target_var, state='readonly')
        self.target_combo.pack(side=tk.LEFT, padx=5)
        
        # 攻击参数框架
        params_frame = tk.Frame(scrollable_frame)
        params_frame.pack(pady=10, fill=tk.X)
        
        # SimBa 参数
        self.simba_frame = tk.LabelFrame(params_frame, text="SimBa 参数")
        self.simba_frame.pack(pady=5, fill=tk.X, padx=5)
        
        # 迭代次数
        iters_frame = tk.Frame(self.simba_frame)
        iters_frame.pack(pady=5, fill=tk.X)
        tk.Label(iters_frame, text="迭代次数:").pack(side=tk.LEFT, padx=5)
        self.iters_var = tk.IntVar(value=1000)
        iters_options = [1000, 2000, 3000, 5000, 10000]
        self.iters_combo = ttk.Combobox(iters_frame, textvariable=self.iters_var, state='readonly', values=iters_options, width=10)
        self.iters_combo.pack(side=tk.LEFT, padx=5)
        
        # Epsilon 参数（最大扰动）
        epsilon_frame = tk.Frame(self.simba_frame)
        epsilon_frame.pack(pady=5, fill=tk.X)
        tk.Label(epsilon_frame, text="最大扰动 (Epsilon):").pack(side=tk.LEFT, padx=5)
        self.epsilon_var = tk.DoubleVar(value=0.2)
        epsilon_options = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.epsilon_combo = ttk.Combobox(epsilon_frame, textvariable=self.epsilon_var, state='readonly', values=epsilon_options, width=10)
        self.epsilon_combo.pack(side=tk.LEFT, padx=5)
        
        # 是否进行目标攻击
        targeted_frame = tk.Frame(self.simba_frame)
        targeted_frame.pack(pady=5, fill=tk.X)
        self.targeted_var = tk.BooleanVar(value=False)
        self.targeted_check = ttk.Checkbutton(targeted_frame, text="目标攻击", variable=self.targeted_var)
        self.targeted_check.pack(side=tk.LEFT, padx=5)
        
        # 源图片选择
        self.source_button = tk.Button(
            scrollable_frame,
            text="选择源图片",
            command=lambda: self.select_adversarial_image('source'),
            width=20,
            height=2
        )
        self.source_button.pack(pady=10)
        
        self.source_label = tk.Label(scrollable_frame, text="未选择源图片", wraplength=400)
        self.source_label.pack(pady=5)
        
        # 目标图片选择（仅用于Boundary Attack）
        self.target_button = tk.Button(
            scrollable_frame,
            text="选择目标图片",
            command=lambda: self.select_adversarial_image('target'),
            width=20,
            height=2
        )
        self.target_button.pack(pady=10)
        
        self.target_label = tk.Label(scrollable_frame, text="未选择目标图片", wraplength=400)
        self.target_label.pack(pady=5)
        
        # 生成对抗样本按钮
        self.generate_button = tk.Button(
            scrollable_frame,
            text="生成对抗样本",
            command=self.generate_adversarial,
            width=20,
            height=2
        )
        self.generate_button.pack(pady=20)
        
        # 进度条框架
        self.progress_frame = tk.Frame(scrollable_frame)
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, padx=5)
        
        # 进度标签
        self.progress_label = tk.Label(self.progress_frame, text="")
        self.progress_label.pack(pady=5)
        
        # 对抗样本预览区域
        self.adv_image_label = tk.Label(scrollable_frame)
        self.adv_image_label.pack(pady=10)
        
        # 显示对抗样本生成结果和进度信息
        self.adv_result_text = tk.Text(scrollable_frame, height=15, width=50)
        self.adv_result_text.pack(pady=20)
        
        # 配置滚动区域
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 绑定攻击方法选择事件
        self.attack_combo.bind('<<ComboboxSelected>>', self.on_attack_method_change)
        
    def on_attack_method_change(self, event=None):
        """当攻击方法改变时更新界面"""
        attack_method = self.attack_var.get()
        if attack_method == "Boundary Attack":
            self.target_button.config(state='normal')
            self.target_label.config(state='normal')
            self.target_combo.config(state='disabled')
            self.simba_frame.pack_forget()  # 隐藏SimBa参数
        else:
            self.target_button.config(state='disabled')
            self.target_label.config(state='disabled')
            self.target_combo.config(state='readonly')
            self.simba_frame.pack(pady=5, fill=tk.X, padx=5)  # 显示SimBa参数
            
    def select_adversarial_image(self, image_type):
        """选择对抗样本生成用的图片"""
        if self.predictor is None:
            messagebox.showerror("错误", "请先选择模型！")
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            if image_type == 'source':
                self.source_label.config(text=file_path)
                self.show_image_preview(file_path, self.adv_image_label)
            else:
                self.target_label.config(text=file_path)
                
    def update_progress(self, step, total_steps):
        """更新进度条"""
        progress = (step / total_steps) * 100
        self.progress_var.set(progress)
        self.progress_label.config(text=f"生成进度: {progress:.1f}%")
        self.window.update()
        
    def update_status(self, text):
        """更新状态文本"""
        self.adv_result_text.insert(tk.END, text + "\n")
        self.adv_result_text.see(tk.END)  # 自动滚动到底部
        self.window.update()
        
    def generate_adversarial(self):
        """生成对抗样本"""
        if self.predictor is None:
            messagebox.showerror("错误", "请先选择模型！")
            return
            
        source_path = self.source_label.cget("text")
        if source_path == "未选择源图片":
            messagebox.showerror("错误", "请先选择源图片！")
            return
            
        try:
            # 禁用生成按钮
            self.generate_button.config(state='disabled')
            
            # 重置进度条和结果文本
            self.progress_var.set(0)
            self.progress_label.config(text="准备生成对抗样本...")
            self.adv_result_text.delete(1.0, tk.END)
            self.window.update()
            
            # 加载和预处理图片
            orig_image = preprocess_image(source_path)
            
            # 根据选择的攻击方法生成对抗样本
            attack_method = self.attack_var.get()
            if attack_method == "Boundary Attack":
                target_path = self.target_label.cget("text")
                if target_path == "未选择目标图片":
                    messagebox.showerror("错误", "请先选择目标图片！")
                    return
                    
                target_image = preprocess_image(target_path)
                # 确保图像在正确的设备上
                orig_image = orig_image.to(self.predictor.device)
                target_image = target_image.to(self.predictor.device)
                
                self.update_status("执行Boundary Attack...")
                adversarial = boundary_attack(
                    self.predictor.model,
                    orig_image,
                    target_image,
                    num_steps=1000,
                    device=self.predictor.device,
                    progress_callback=self.update_progress
                )
            else:  # SimBa
                # 获取原始图片的预测类别
                with torch.no_grad():
                    orig_output = self.predictor.model(orig_image.unsqueeze(0).to(self.predictor.device))
                    orig_pred = orig_output.argmax(dim=1).item()
                
                # 如果是目标攻击，检查是否选择了目标类别
                if self.targeted_var.get():
                    target_class = self.target_combo.current()
                    if target_class < 0:
                        messagebox.showerror("错误", "请选择目标类别！")
                        return
                else:
                    target_class = orig_pred  # 使用原始类别
                    
                # 获取参数
                epsilon = self.epsilon_var.get()
                num_iters = self.iters_var.get()
                targeted = self.targeted_var.get()
                
                self.update_status(f"执行SimBa...\n参数: epsilon={epsilon}, 迭代次数={num_iters}, 目标攻击={targeted}")
                
                # 创建SimBA攻击器
                from kitti_simba_attack import KittiSimBA
                attacker = KittiSimBA(self.predictor.model)
                
                # 定义进度回调函数
                def update_progress(step, message):
                    # 更新进度条（使用平滑过渡）
                    progress = (step / num_iters) * 100
                    current_progress = self.progress_var.get()
                    if progress > current_progress:
                        # 使用线性插值进行平滑过渡
                        steps = 10
                        for i in range(steps):
                            smooth_progress = current_progress + (progress - current_progress) * (i + 1) / steps
                            self.progress_var.set(smooth_progress)
                            self.progress_label.config(text=f"生成进度: {smooth_progress:.1f}%")
                            self.window.update()
                            self.window.after(10)  # 添加小延迟使动画更流畅
                    
                    # 更新结果文本
                    self.update_status(message)
                
                # 执行攻击
                adversarial = attacker.simba_single(
                    orig_image.squeeze(0),
                    torch.tensor([target_class], device=self.predictor.device),
                    num_iters=num_iters,
                    epsilon=epsilon,
                    targeted=targeted,
                    progress_callback=update_progress
                )
            
            # 保存对抗样本
            adv_img = postprocess_image(adversarial)
            adv_img.save("adversarial_sample.png")
            
            # 显示对抗样本
            self.show_image_preview("adversarial_sample.png", self.adv_image_label)
            
            # 显示预测结果
            with torch.no_grad():
                orig_pred = self.predictor.model(orig_image.unsqueeze(0).to(self.predictor.device)).argmax(dim=1).item()
                adv_pred = self.predictor.model(adversarial.unsqueeze(0).to(self.predictor.device)).argmax(dim=1).item()
                
                # 计算置信度
                orig_probs = F.softmax(self.predictor.model(orig_image.unsqueeze(0).to(self.predictor.device)), dim=1)
                adv_probs = F.softmax(self.predictor.model(adversarial.unsqueeze(0).to(self.predictor.device)), dim=1)
                
                # 计算扰动大小
                perturbation = torch.norm((adversarial - orig_image).view(-1), p=2).item()
                
            result_text = f"源图片预测类别: {self.predictor.classes[orig_pred]} (置信度: {orig_probs[0, orig_pred]:.4f})\n"
            result_text += f"对抗样本预测类别: {self.predictor.classes[adv_pred]} (置信度: {adv_probs[0, adv_pred]:.4f})\n"
            
            if attack_method == "SimBa":
                result_text += f"目标类别置信度: {adv_probs[0, target_class]:.4f}\n"
                result_text += f"扰动大小(L2范数): {perturbation:.6f}\n"
                
                if adv_pred == target_class:
                    result_text += "\n攻击成功! 对抗样本被成功分类为目标类别。"
                else:
                    result_text += "\n攻击未成功。可以尝试增加epsilon或迭代次数。"
            
            self.adv_result_text.delete(1.0, tk.END)
            self.adv_result_text.insert(tk.END, result_text)
            
            # 更新进度条为完成状态
            self.progress_var.set(100)
            self.progress_label.config(text="生成完成！")
            self.update_status("对抗样本生成完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"生成对抗样本时出错：{str(e)}")
            self.update_status(f"错误: {str(e)}")
        finally:
            # 重新启用生成按钮
            self.generate_button.config(state='normal')
            
    def show_image_preview(self, image_path, label):
        """显示图片预览"""
        try:
            # 加载图片并调整大小
            image = Image.open(image_path)
            # 计算调整后的尺寸，保持宽高比
            display_size = (400, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # 更新图片显示
            label.config(image=photo)
            label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"加载图片预览时出错: {str(e)}")
            
    def on_model_select(self, event=None):
        """当选择模型时更新预测器"""
        model_name = self.model_var.get()
        if model_name == 'VGG16':
            model_path = 'best_vgg_model.pth'
        elif model_name == 'ResNet50':
            model_path = 'best_resnet_model.pth'
        elif model_name == 'MobileNetV2':
            model_path = 'best_mobilenet_model.pth'
        else:
            return
            
        try:
            if os.path.exists(model_path):
                self.predictor = KITTIPredictor(model_path)
                # 更新目标类别下拉框
                self.target_combo['values'] = self.predictor.classes
                messagebox.showinfo("成功", f"已加载{model_name}模型")
            else:
                messagebox.showerror("错误", f"未找到模型文件: {model_path}")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型时出错: {str(e)}")
            
    def select_image(self):
        """选择图片文件"""
        if self.predictor is None:
            messagebox.showerror("错误", "请先选择模型！")
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self.path_label.config(text=file_path)
            self.show_image_preview(file_path, self.image_label)
            
    def predict_image(self):
        """预测选中的图片"""
        if self.predictor is None:
            messagebox.showerror("错误", "请先选择模型！")
            return
            
        image_path = self.path_label.cget("text")
        if image_path == "未选择图片":
            messagebox.showerror("错误", "请先选择图片！")
            return
            
        try:
            # 进行预测
            predicted_class, confidence, all_probabilities = self.predictor.predict(image_path)
            
            # 显示预测结果
            result_text = f"预测类别: {predicted_class}\n"
            result_text += f"置信度: {confidence:.2%}\n\n"
            result_text += "各类别概率:\n"
            
            for i, (cls, prob) in enumerate(zip(self.predictor.classes, all_probabilities)):
                result_text += f"{cls}: {prob:.2%}\n"
                
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            
            # 更新概率条形图
            self.ax.clear()
            self.ax.bar(self.predictor.classes, all_probabilities)
            self.ax.set_title('各类别预测概率', fontsize=12)
            self.ax.set_xlabel('类别', fontsize=10)
            self.ax.set_ylabel('概率', fontsize=10)
            plt.xticks(rotation=45, fontsize=8)
            plt.yticks(fontsize=8)
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("错误", f"预测过程中出错：{str(e)}")
            
    def run(self):
        """运行GUI程序"""
        self.window.mainloop()

def main():
    try:
        # 检查模型文件是否存在
        if not os.path.exists('best_vgg_model.pth') and not os.path.exists('best_resnet_model.pth') and not os.path.exists('best_mobilenet_model.pth'):
            print("错误：未找到任何模型文件")
            return
            
        # 创建并运行GUI
        gui = PredictorGUI()
        gui.run()
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")

if __name__ == '__main__':
    main() 