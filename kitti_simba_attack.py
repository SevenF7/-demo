import torch
import torch.nn.functional as F
import torchvision.transforms as trans
import numpy as np
from scipy.fftpack import dct, idct

# KITTI数据集的均值和标准差
KITTI_SIZE = 224  # 使用224x224作为输入大小
KITTI_MEAN = [0.485, 0.456, 0.406]  # 使用ImageNet的均值和标准差，因为KITTI数据集通常使用这些值
KITTI_STD = [0.229, 0.224, 0.225]

KITTI_TRANSFORM = trans.Compose([
    trans.Resize((224, 224)),
    trans.ToTensor(),
    trans.Normalize(mean=KITTI_MEAN, std=KITTI_STD)
])

def apply_normalization(imgs):
    """对KITTI图像进行标准化"""
    mean = torch.tensor(KITTI_MEAN).view(1, 3, 1, 1).cuda()
    std = torch.tensor(KITTI_STD).view(1, 3, 1, 1).cuda()
    return (imgs - mean) / std

def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=0.0):
    """对图像块进行IDCT变换"""
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[i, :, :int(block_size * ratio[i]), :int(block_size * ratio[i])] = 1
    else:
        mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].numpy()
            if masked:
                submat = submat * mask
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = torch.from_numpy(idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho'))
    if linf_bound > 0:
        return z.clamp(-linf_bound, linf_bound)
    else:
        return z

class KittiSimBA:
    def __init__(self, model, image_size=224):
        self.model = model
        self.image_size = image_size
        self.model.eval()
    
    def expand_vector(self, x, size):
        """扩展向量到指定大小"""
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z
    
    def get_probs(self, x, y):
        """获取模型对输入x的预测概率"""
        x = x.cuda()  # 将输入移到GPU
        output = self.model(x)
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y.cuda())  # 确保y也在GPU上
        return torch.diag(probs)
    
    def get_preds(self, x):
        """获取模型对输入x的预测结果"""
        x = x.cuda()  # 将输入移到GPU
        output = self.model(x)
        _, preds = output.data.max(1)
        return preds

    def simba_single(self, x, y, num_iters=10000, epsilon=0.2, targeted=False, progress_callback=None):
        """对单张图像进行SimBA攻击"""
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0).cuda()  # 将x移到GPU
        y = y.cuda()  # 将y移到GPU
        last_prob = self.get_probs(x, y)
        
        for i in range(num_iters):
            diff = torch.zeros(n_dims).cuda()  # 将diff移到GPU
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if targeted != (right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if i % 100 == 0:
                if progress_callback:
                    progress_callback(i, f'Iteration {i}: probability = {last_prob.item():.4f}')
                else:
                    print(f'Iteration {i}: probability = {last_prob.item():.4f}')
        return x.squeeze().cpu()  # 返回CPU上的结果

    def simba_batch(self, images_batch, labels_batch, max_iters=10000, freq_dims=8, stride=1, epsilon=0.2, 
                    linf_bound=0.0, order='rand', targeted=False, pixel_attack=False, log_every=100):
        """对批量图像进行SimBA攻击"""
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)
        assert self.image_size == image_size
        
        # 将输入移到GPU
        images_batch = images_batch.cuda()
        labels_batch = labels_batch.cuda()
        
        # 生成坐标顺序
        if order == 'rand':
            indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
        else:
            indices = torch.randperm(3 * image_size * image_size)[:max_iters]
            
        expand_dims = freq_dims if order == 'rand' else image_size
        n_dims = 3 * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims).cuda()  # 将x移到GPU
        
        # 记录攻击过程
        probs = torch.zeros(batch_size, max_iters).cuda()  # 将probs移到GPU
        succs = torch.zeros(batch_size, max_iters).cuda()  # 将succs移到GPU
        queries = torch.zeros(batch_size, max_iters).cuda()  # 将queries移到GPU
        l2_norms = torch.zeros(batch_size, max_iters).cuda()  # 将l2_norms移到GPU
        linf_norms = torch.zeros(batch_size, max_iters).cuda()  # 将linf_norms移到GPU
        
        prev_probs = self.get_probs(images_batch, labels_batch)
        preds = self.get_preds(images_batch)
        
        if pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: block_idct(z, block_size=image_size, linf_bound=linf_bound)
            
        remaining_indices = torch.arange(0, batch_size).long().cuda()  # 将remaining_indices移到GPU
        
        for k in range(max_iters):
            dim = indices[k]
            expanded = (images_batch[remaining_indices] + trans(self.expand_vector(x[remaining_indices], expand_dims))).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims))
            
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            
            preds_next = self.get_preds(expanded)
            preds[remaining_indices] = preds_next
            
            if targeted:
                remaining = preds.ne(labels_batch)
            else:
                remaining = preds.eq(labels_batch)
                
            if remaining.sum() == 0:
                adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
                probs_k = self.get_probs(adv, labels_batch)
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k).cuda()
                queries[:, k:] = torch.zeros(batch_size, max_iters - k).cuda()
                break
                
            remaining_indices = torch.arange(0, batch_size)[remaining].long().cuda()
            if k > 0:
                succs[:, k] = ~remaining
                
            diff = torch.zeros(remaining.sum(), n_dims).cuda()  # 将diff移到GPU
            diff[:, dim] = epsilon
            
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            
            # 尝试负方向
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(left_vec, expand_dims))).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size).cuda()  # 将queries_k移到GPU
            queries_k[remaining_indices] += 1
            
            if targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                improved = left_probs.lt(prev_probs[remaining_indices])
                
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved]] += 1
                
            # 尝试正方向
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(right_vec, expand_dims))).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            
            if targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
                
            probs_k = prev_probs.clone()
            
            # 更新x
            if improved.sum() > 0:
                left_indices = remaining_indices[improved]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
                
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
                
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            
            if (k + 1) % log_every == 0 or k == max_iters - 1:
                print(f'Iteration {k + 1}: queries = {queries.sum(1).mean():.4f}, prob = {probs[:, k].mean():.4f}, remaining = {remaining.float().mean():.4f}')
                
        expanded = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
        preds = self.get_preds(expanded)
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        succs[:, max_iters-1] = ~remaining
        
        # 返回CPU上的结果
        return expanded.cpu(), probs.cpu(), succs.cpu(), queries.cpu(), l2_norms.cpu(), linf_norms.cpu() 