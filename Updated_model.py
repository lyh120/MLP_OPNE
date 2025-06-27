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




def deduplicate_data(settings, qs, tol=1e-3):
    """对历史设置去重。tol为相似判据"""
    settings = np.array(settings)
    qs = np.array(qs)
    keep = []
    seen = []
    for idx, s in enumerate(settings):
        duplicate = False
        for s2 in seen:
            if np.linalg.norm(s - s2) < tol:  # 如果距离过近，视为重复
                duplicate = True
                break
        if not duplicate:
            seen.append(s)
            keep.append(idx)
    return settings[keep], qs[keep]


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
                              num_samples=4,
                              epochs=50,
                              lr=1e-4,
                              Lip_lambda=0.2,
                              L_const=np.ones([5,12]),
                              setting_list=None,
                              q_list=None,
                              save_path='enhanced_model.pth'):
    """改进后的在线微调函数"""
    # # 直接使用完整的 `setting_list` 和 `q_list` 列表进行微调，不再进行子集选择
    # setting_list = np.array(setting_list)
    # q_list = np.array(q_list)
    #上述方案有误，应该使用，最近的最新加入的4组数据进行在线微调

    #整合为二维数组
    setting_list = setting_list.reshape(-1,12)
    q_list = q_list.reshape(-1,5)

    #条件判断提高安全性：数据不足时返回原来的模型
    if setting_list.shape[0]<num_samples:
        print("警告：提供的数据样本数量不足4组，无法进行在线微调")
        return model
    # 滑动窗口数据增强
    def sliding_window_augmentation(data, window_size=3):
        augmented = []
        for i in range(len(data)-window_size+1):
            window = data[i:i+window_size]
            augmented.append(np.mean(window, axis=0))
            augmented.append(window[1] + np.random.normal(0, 0.01, window[1].shape))
        return np.array(augmented)
    # 不断使用最近，最新加入的4组数据进行增强
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
    
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=10,
    #     T_mult=1,
    #     eta_min=1e-6
    # )
    # 使用StepLR基础调度器（每5个epoch衰减为原来0.8倍）
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,  # 每5个epoch调整一次
        gamma=0.8      # 学习率衰减系数
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


def multi_strategy_search(current_best, cons, bounds, model, 
                         n_candidates=200, 
                         n_rounds=3, 
                         scaler_X=None):
    """
    多策略协同全局搜索
    - current_best: 当前最优参数
    - cons: 可用的约束函数
    - bounds: 参数边界
    - model: 当前的神经网络模型
    - scaler_X: 用于数据标准化
    """
    dim = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    all_candidates = []
    
    ## 1. LHS全局采样
    N1 = n_candidates // 3
    sampler = qmc.LatinHypercube(d=dim)
    samples = sampler.random(N1)
    lhs_points = qmc.scale(samples, lower_bounds, upper_bounds)
    all_candidates.extend(lhs_points)
    
    ## 2. 遗传变异(从最优点引入噪声+部分变量大跳变)
    N2 = n_candidates // 3
    for _ in range(N2):
        mutant = current_best.copy()
        idx = np.random.choice(dim, max(1, dim//6), replace=False)
        mutant[idx] = lower_bounds[idx] + np.random.rand(len(idx)) * (upper_bounds[idx]-lower_bounds[idx])
        # 加点全局扰动
        mutant += np.random.normal(0, 0.2, dim) * (upper_bounds-lower_bounds)
        mutant = np.clip(mutant, lower_bounds, upper_bounds)
        all_candidates.append(mutant)
    
    ## 3. 贝叶斯采样 (利用Gaussian Process)
    N3 = n_candidates - N1 - N2
    # 采历史点、随机扰动为输入，训练一个简单GP
    # 用NN模型特征+Q结果拟合
    # 假设我们有部分历史点
    if hasattr(model, 'scaler_X') and scaler_X is not None:
        # 这里可以用模型自己的历史点，也可传入参数
        pass  # 暂略
    # 暂用随机点替代
    for _ in range(N3):
        # 简单利用模型预测：全局最小化负分数
        def obj(x):
            x = np.array(x).reshape(1, -1)
            x_scaled = np.zeros_like(x)
            for i, (lb, ub) in enumerate(bounds):
                x_scaled[..., i] = (x[..., i] - lb) / (ub - lb)
            with torch.no_grad():
                tensor = torch.FloatTensor(x_scaled)
                score = model(tensor).sum().item()
            return -score
        x0 = lower_bounds + np.random.rand(dim) * (upper_bounds-lower_bounds)
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter':10})
        all_candidates.append(res.x)
        
    ## 合并候选, 取整 & 排重 & 过滤
    candidates = np.vstack(all_candidates)
    candidates = np.clip(candidates, lower_bounds, upper_bounds)
    # 值变成一位小数，因为原来是这样round的
    candidates = np.round(candidates, 1)
    unique_candidates = np.unique(candidates, axis=0)
    # 过滤约束
    valid_points = [p for p in unique_candidates if cons(p) == 1]
    print(f"多策略采样候选: {len(valid_points)}/{len(unique_candidates)} 满足约束")
    # 预测分数，选最优
    if not valid_points:
        print("无可用候选，返回当前最优")
        return current_best
    points_scaled = np.zeros_like(valid_points)
    for i, (lb, ub) in enumerate(bounds):
        points_scaled[:, i] = (np.array(valid_points)[:, i] - lb) / (ub - lb)
    with torch.no_grad():
        tensor = torch.FloatTensor(points_scaled)
        scores = model(tensor).sum(dim=1).numpy()
    best_idx = np.argmax(scores)
    return np.array(valid_points[best_idx])


def online_opt(x0, iter_num=30, seed=42, early_stop=True, patience=5, 
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

    global_search_flag = False
    global_search_count = 0
    max_global_search_rounds = 10
    degrade_threshold = 1  # 只要一次剧烈下降就触发
    # 记录峰值
    best_solution = x0.copy()
    best_fitness = None
    last_fitness = None
    best_Q = None
    model_snapshots = []
    settings = []
    qs = []
    random.seed(seed)
    np.random.seed(seed)

    # --- 历史数据记录 ---
    apply_parallel(x0)
    q0 = query_parallel(avg_times=3)
    settings.append(np.array(x0))
    qs.append(q0)
    best_fitness = np.mean(q0)
    best_Q = np.array(q0)
    no_improvement_count = 0

    fitness_history = [best_fitness]
    q_min_history = [np.min(q0)]
    q_avg_history = [np.mean(q0)]

    print("初始化已完成，开始迭代优化:")

    start = time.time()

    for index in range(iter_num):
        # ----- 峰值剧烈下降检测 -----
        degrade_flag = False
        if last_fitness is not None and fitness_history[-1] < 0.98 * last_fitness:
            degrade_flag = True
            print(f"!!! 监测到剧烈下降：{fitness_history[-1]:.3f} < {last_fitness:.3f}")

        # 回退并且进入全局搜索
        if (not global_search_flag) and (degrade_flag or no_improvement_count >= patience):
            if model_snapshots:
                # 恢复到最后一次模型快照与峰值解
                print("===> 回退到历史峰值，转入多策略协同搜索！")
                model.load_state_dict(model_snapshots[-1]['model'])
                model.scaler_X = model_snapshots[-1]['scaler_X']
                model.scaler_y = model_snapshots[-1]['scaler_y']
                best_solution = model_snapshots[-1]['best_solution']
                best_fitness = model_snapshots[-1]['best_fitness']
                # 重新应用并测得真实Q
                apply_parallel(best_solution)
                best_Q = query_parallel(avg_times=3)
                settings.append(best_solution.copy())
                qs.append(best_Q.copy())
            else:
                print("提示：没有快照可回退。")
            # ---- 去重历史数据 ----
            settings, qs = deduplicate_data(settings, qs)
            print(f"去重后样本数:{len(settings)}")
            # 转入全局多策略阶段
            global_search_flag = True
            global_search_count = 0
            no_improvement_count = 0
            continue

        # --- 采样新解 ---
        cons = resolution_shape(max_gradient, settings, qs)
        if not global_search_flag:
            # 正常局部+finetune阶段
            new_solution = generate_best_solution(
                settings, qs, cons, bounds, model, noise_scale=0.05)
        else:
            # 多策略全局阶段
            new_solution = multi_strategy_search(
                current_best=best_solution, cons=cons, bounds=bounds, 
                model=model, scaler_X=model.scaler_X)
            global_search_count += 1
        # --- 应用并评估 ---
        apply_parallel(new_solution)
        current_Q = query_parallel(avg_times=3)
        fitness = np.mean(current_Q)
        settings.append(new_solution.copy())
        qs.append(current_Q.copy())
        fitness_history.append(fitness)
        q_min_history.append(np.min(current_Q))
        q_avg_history.append(np.mean(current_Q))

        # ---- 峰值/最优更新 ----
        if fitness > best_fitness:
            best_solution = new_solution.copy()
            best_fitness = fitness
            best_Q = current_Q.copy()
            no_improvement_count = 0
            # 快照保存
            snapshot = {'model': model.state_dict(),
                        'scaler_X': model.scaler_X,
                        'scaler_y': model.scaler_y,
                        'best_solution': best_solution.copy(),
                        'best_fitness': best_fitness}
            model_snapshots.append(snapshot)
            if len(model_snapshots) > 5:
                model_snapshots.pop(0)
        else:
            no_improvement_count += 1

        last_fitness = fitness

        # --- 微调阶段 ---
        if not global_search_flag and ((index + 1) % 3 == 0):
            # 数据去重，增强再finetune
            settings_dedup, qs_dedup = deduplicate_data(settings, qs)
            model = adaptive_online_finetuning(
                model=model, setting_list=settings_dedup, q_list=qs_dedup)

        # ---- 多策略全局阶段停止准则 ----
        if global_search_flag and global_search_count >= max_global_search_rounds and no_improvement_count >= patience:
            print("多策略全局搜索终止，未获得新突破")
            break

        # --- 每轮保存 ---
        np.savez(save_name+'.npz', 
                 qs=np.array(qs), 
                 settings=np.array(settings), 
                 best_setting=np.array(best_solution), 
                 best_q=np.squeeze(best_Q))
        
        # --- 输出 ---
        print(f"Iter {index:02d} | F:{fitness:.4f} | Best:{best_fitness:.4f} | Imprv_cnt: {no_improvement_count}")

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
    #x = [16, 25, 18, 10, 25, 18, -1, -1, -1, -1, -1, -1]
    x = [18.8 ,29. , 18.7, 14.6 ,27.7 ,21.7 ,-0.1, -0.7 ,-0.4 ,-0.7 ,-1.2 ,-1.7]
    online_opt(x0=x, iter_num=50, seed=42, early_stop=True, patience=10, 
               max_gradient=Lip, save_name='test')
