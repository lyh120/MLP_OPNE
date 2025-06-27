import numpy as np
import time, random
from utilities.utils import apply_parallel, query_parallel, query_setting
import torch
import scipy.stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import qmc

class LipschitzNN(nn.Module):
    def __init__(self, L_lambda, L_const, input_dim=12, output_dim=5):
        super().__init__()
        self.Net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, output_dim)
        )
        self.lip_lambda = L_lambda
        self.scaler_X = None
        self.scaler_y = None
        self.Lip_const = torch.FloatTensor(L_const)
    
    def forward(self, x):
        return self.Net(x)
    
    def lip_loss(self, x, y_pred):
        sigma_x = torch.FloatTensor(self.scaler_X.scale_)
        sigma_y = torch.FloatTensor(self.scaler_y.scale_)
        Lip_scaled = self.Lip_const * (sigma_x.reshape(1, -1) / sigma_y.reshape(-1, 1))
        batch_size = x.size(0)
        
        jacobian = torch.zeros(batch_size, y_pred.size(1), x.size(1)).to(x.device)
        for i in range(y_pred.size(1)):
            gradients = torch.autograd.grad(
                outputs=y_pred[:, i].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True
            )[0]
            jacobian[:, i, :] = gradients
        
        jacobian_abs = torch.abs(jacobian)
        Lip_expanded = Lip_scaled.unsqueeze(0).expand(batch_size, -1, -1)
        excess = (jacobian_abs - Lip_expanded).clamp(min=0)
        return torch.mean(excess**2)

def resolution_shape(max_gradient, settings, qs):
    mg = np.array(max_gradient)
    
    def constraint(x_tar):
        for set, q in zip(settings, qs):
            delta = abs(np.array(set) - x_tar)
            degrade = np.dot(mg, delta)
            q_pred = q - degrade
            if min(q_pred) > 9:
                return 1
        return 0
    return constraint

def generate_best_solution(settings, qs, cons, bounds, model, 
                          batch_size=100, noise_scale=0.05, noise_increment=0.1,
                          target_points=500, max_iters=100):
    """改进后的生成最优解函数"""
    def _scale_params(x, bounds):
        x = np.array(x)
        x_norm = np.zeros_like(x, dtype=float)
        for i, (lb, ub) in enumerate(bounds):
            x_norm[..., i] = (x[..., i] - lb) / (ub - lb)
        return x_norm

    def _model_predict(model, inputs):
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(inputs)
            preds = model(inputs_tensor).sum(dim=1)
            return preds.numpy()

    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    dim = len(bounds)
    
    # 混合采样参数
    lhs_ratio = 0.7
    current_point = settings[-1]

    satisfied_points = []
    for _ in range(max_iters):
        all_points = []
        
        # 只围绕当前最优解生成候选
        center = current_point
        local_lb = np.clip(center - noise_scale*(upper_bounds - lower_bounds), lower_bounds, upper_bounds)
        local_ub = np.clip(center + noise_scale*(upper_bounds - lower_bounds), lower_bounds, upper_bounds)
        
        # LHS采样（局部范围）
        lhs_samples = qmc.LatinHypercube(d=dim).random(n=int(batch_size*lhs_ratio))
        lhs_samples = qmc.scale(lhs_samples, local_lb, local_ub)
        
        # 高斯采样
        gauss_samples = center + np.random.normal(
            scale=noise_scale, 
            size=(int(batch_size*(1-lhs_ratio)), dim)
        ) * (upper_bounds - lower_bounds)
        
        # 合并并处理
        candidates = np.vstack([lhs_samples, gauss_samples])
        candidates = np.clip(candidates, lower_bounds, upper_bounds)
        candidates = np.round(candidates, 1)
        all_points.extend(candidates)

        # 强制包含当前解
        all_points.append(current_point)
        
        # 过滤满足约束的点
        new_valid = [pt for pt in all_points if cons(pt) == 1]
        satisfied_points.extend(new_valid)
        
        if len(satisfied_points) >= target_points:
            break

    if not satisfied_points:
        print("无有效候选，返回当前解")
        return current_point
    
    candidates_scaled = _scale_params(satisfied_points, bounds)
    scores = _model_predict(model, candidates_scaled)
    best_idx = np.argmax(scores)
    best_solution = satisfied_points[best_idx]
    
    current_score = _model_predict(model, _scale_params([current_point], bounds))[0]
    if scores[best_idx] > current_score:
        print(f"找到更优解：{scores[best_idx]:.2f} > {current_score:.2f}")
        return best_solution
    else:
        new_noise = noise_scale * (1 + noise_increment)
        print(f"未找到改进，扩大噪声至：{new_noise:.3f}")
        return generate_best_solution(
            settings=np.vstack([settings, best_solution]),
            qs=np.vstack([qs, qs[-1]]),
            cons=cons,
            bounds=bounds,
            model=model,
            noise_scale=new_noise,
            noise_increment=noise_increment,
            target_points=target_points,
            max_iters=max_iters
        )

def adaptive_online_finetuning(model=None, 
                              num_samples=5,
                              epochs=50,
                              lr=1e-4,
                              Lip_lambda=0.2,
                              L_const=np.ones([5,12]),
                              setting_list=None,
                              q_list=None,
                              save_path='enhanced_model.pth'):
    """改进后的在线微调函数"""
    setting_list = setting_list.reshape(-1, 12)
    q_list = q_list.reshape(-1, 5)
    
    # 新增条件判断：数据不足时直接返回
    if setting_list.shape[0] < 4:
        print("数据不足4组，跳过微调")
        return model

    # 滑动窗口数据增强
    def sliding_window_augmentation(data, window_size=3):
        augmented = []
        for i in range(len(data)-window_size+1):
            window = data[i:i+window_size]
            augmented.append(np.mean(window, axis=0))
            augmented.append(window[1] + np.random.normal(0, 0.01, window[1].shape))
        return np.array(augmented)

    # 使用最近4组数据进行增强
    raw_configs = setting_list[-4:]
    raw_q = q_list[-4:]
    
    aug_configs = sliding_window_augmentation(raw_configs)
    aug_q = sliding_window_augmentation(raw_q)

    real_configs = np.vstack([raw_configs, aug_configs])
    real_q = np.vstack([raw_q, aug_q])

    # 标准化处理
    real_configs_scaled = model.scaler_X.transform(real_configs)
    real_q_scaled = model.scaler_y.transform(real_q)

    # 创建数据加载器
    dataset = TensorDataset(
        torch.FloatTensor(real_configs_scaled),
        torch.FloatTensor(real_q_scaled))
    loader = DataLoader(dataset, 
                      batch_size=min(8, len(dataset)),
                      shuffle=True,
                      pin_memory=True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,
        T_mult=1,
        eta_min=1e-6
    )
    
    best_loss = float('inf')
    patience = 8
    patience_counter = 0
    losses = []
    grad_magnitudes = []
    
    print(f"\n=== 开始微调 (增强后样本数: {len(dataset)}) ===")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_grad = 0.0
        
        for x, y in loader:
            x.requires_grad_(True)
            
            optimizer.zero_grad()
            
            pred = model(x)
            
            mse_loss = nn.MSELoss()(pred, y)
            lip_loss = model.lip_loss(x, pred)
            total_loss = mse_loss + model.lip_lambda * lip_loss
            
            total_loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                norm_type=2
            )
            
            total_grad += grad_norm.item()
            
            optimizer.step()
            epoch_loss += total_loss.item()
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(loader)
        avg_grad = total_grad / len(loader)
        losses.append(avg_loss)
        grad_magnitudes.append(avg_grad)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"fine_tuned_best.pth")
        else:
            patience_counter += 1
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Loss: {avg_loss:.3e} | "
              f"Grad: {avg_grad:.2f} | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"早停触发，最佳loss: {best_loss:.4e}")
            break
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_X': model.scaler_X,
        'scaler_y': model.scaler_y
    }
    torch.save(checkpoint, f"enhanced_model.pth")    
    return model

def online_opt(x0, iter_num=20, seed=42, early_stop=True, patience=5, 
               Lip_lambda=0.2, max_gradient=np.ones([5,12]), save_name='test',
               model_path='enhanced_model.pth'):
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到预训练模型 {model_path}")

    model = LipschitzNN(L_lambda=Lip_lambda, L_const=max_gradient)
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.scaler_X = checkpoint['scaler_X']
        model.scaler_y = checkpoint['scaler_y']
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}") from e

    print("\n=== 成功加载增强模型 ===")

    bounds = [(14, 23), (22, 30), (16, 22), (8, 15), (22,30), 
             (16, 22),(-2, 0), (-2, 0), (-2, 0), (-2, 0), (-2, 0), (-2, 0)]
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]

    random.seed(seed)
    np.random.seed(seed)
    
    best_solution = x0
    no_improvement_count = 0
    
    settings = []
    qs = []
    apply_parallel(x0)
    settings.append(np.array(x0))
    q0 = query_parallel(avg_times=3)
    qs.append(q0)
    best_fitness = np.mean(q0)
    best_Q = np.array(q0)
    L_const = max_gradient
    
    fitness_history = []
    q_min_history = []
    q_avg_history = []
    
    start = time.time()
    
    for index in range(iter_num):
        cons = resolution_shape(L_const, settings, qs)
        
        new_solution = generate_best_solution(settings, qs, cons, bounds, model)
        
        apply_parallel(new_solution)
        current_Q = query_parallel()

        settings = np.concatenate((settings, new_solution.reshape(1,-1)), axis=0)
        qs = np.concatenate((qs, current_Q.reshape(1,-1)), axis=0)
        
        # 每积累4组数据才进行微调
        if (index + 1) % 4 == 0:
            model = adaptive_online_finetuning(
                model=model,
                setting_list=settings,
                q_list=qs,
                save_path='enhanced_model.pth'
            )
        
        current_fitness_value = np.mean(current_Q)
        print(f"Iteration {index}: Current Fitness = {current_fitness_value}")

        if current_fitness_value > best_fitness:
            best_solution = new_solution
            best_fitness = current_fitness_value
            best_Q = current_Q
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        fitness_history.append(current_fitness_value)
        q_min_history.append(np.min(current_Q))
        q_avg_history.append(np.mean(current_Q))
        
        np.savez(save_name+'.npz', qs=np.array(qs), settings=np.array(settings), 
                best_setting=np.array(best_solution), best_q=np.squeeze(best_Q))
        
        if early_stop and (no_improvement_count >= patience):
            print(f"Early stopping at Iteration {index}")
            break

    # 绘图代码保持不变
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fitness_history, label='Average Fitness')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Fitness over Iterations')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(q_min_history, label='Minimum Q Value')
    plt.plot(q_avg_history, label='Average Q Value')
    plt.xlabel('Iteration')
    plt.ylabel('Q Value')
    plt.title('Q Value over Iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()
        
    end = time.time()
    print(f"Optimal Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}, Q Values: {best_Q}")
    print(f"Total Time: {end - start:.2f}s")
    
if __name__=='__main__':
    Lip_data = np.load('./Lip_GN_5wave_0319.npz')
    Lip = Lip_data['mgn_list'][-1]
    #x = [16,25,18,10,25,18,-1,-1,-1,-1,-1,-1]
    x = [18.8 ,29. , 18.7, 14.6 ,27.7 ,21.7 ,-0.1, -0.7 ,-0.4 ,-0.7 ,-1.2 ,-1.7]
    online_opt(x0=x, iter_num=40, seed=42, early_stop=True, patience=10, 
               max_gradient=Lip, save_name='test')