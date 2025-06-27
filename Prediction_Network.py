import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score

# ----------------------
# 1. 增强型模型定义
# ----------------------
class EnhancedLipschitzNN(nn.Module):
    def __init__(self, input_dim=12, output_dim=6):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.predictor = nn.Sequential(
            nn.Linear(32, 24),
            nn.GELU(),
            nn.Linear(24, output_dim)
        )
        self.lip_lambda = 0.05
        self.scaler_X = None
        self.scaler_y = None

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.predictor(features)

    def lip_loss(self, x, y_pred):
        if x.grad is not None:
            x.grad.zero_()
        gradients = torch.autograd.grad(
            outputs=y_pred, inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True, retain_graph=True,
            allow_unused=False
        )[0]
        grad_norm = torch.norm(gradients, p=2, dim=1)
        return torch.mean((grad_norm - 1.0).clamp(min=0)**2)

# ----------------------
# 可视化工具函数
# ----------------------
def plot_loss_curves(train_losses, val_losses, title="Training Process"):
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(y_true, y_pred, title="Prediction Comparison"):
    plt.figure(figsize=(15, 6))
    for i in range(y_true.shape[1]):
        plt.subplot(2, 3, i+1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel(f'True Q{i+1}')
        plt.ylabel(f'Predicted Q{i+1}')
        plt.title(f'Q{i+1} Prediction (R2={r2_score(y_true[:,i], y_pred[:,i]):.3f})')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

# ----------------------
# 2. 高精度离线训练
# ----------------------
def enhanced_offline_training(save_path="enhanced_model.pth"):
    # 数据加载与预处理
    loaded_data = np.load("C:\\Users\\55266\\Desktop\\论文综述\\new_dataset2.npz")
    setting_list = loaded_data['setting_list'].reshape(-1, 12)
    q_list = loaded_data['q_list'].reshape(-1, 6)

    # 数据增强
    def add_noise(data, noise_level=0.01):
        return data + np.random.normal(0, noise_level, data.shape)
    
    # 扩展数据集
    augmented_settings = np.vstack([setting_list, add_noise(setting_list)])
    augmented_q = np.vstack([q_list, q_list])

    # 标准化处理
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(augmented_settings)
    y_scaled = scaler_y.fit_transform(augmented_q)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # 数据加载器
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=128
    )

    # 模型初始化
    model = EnhancedLipschitzNN()
    model.scaler_X = scaler_X
    model.scaler_y = scaler_y
    
    # 优化配置
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    criterion = nn.MSELoss()

    # 训练循环
    best_loss = float('inf')
    train_losses, val_losses = [], []
    lr_history = []
    
    print("\n=== 开始增强型离线训练 ===")
    for epoch in range(200):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch.requires_grad_(True)
            optimizer.zero_grad()
            
            pred = model(X_batch)
            loss = criterion(pred, y_batch) + model.lip_lambda * model.lip_loss(X_batch, pred)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
        
        # 记录学习率
        lr_history.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        
        # 保存最佳模型
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1}: 保存最佳模型 (Val Loss: {best_loss:.4f})")
        
        # 记录损失
        train_losses.append(epoch_loss/len(train_loader))
        val_losses.append(avg_val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/200 | Train Loss: {train_losses[-1]:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 可视化训练过程
    plot_loss_curves(train_losses, val_losses)
    
    # 学习率变化曲线
    plt.figure(figsize=(10,4))
    plt.plot(lr_history)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.show()

    # 测试集评估
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.FloatTensor(X_test))
        y_pred = model.scaler_y.inverse_transform(y_pred_scaled.numpy())
        y_test_unscaled = model.scaler_y.inverse_transform(y_test)
        
        # 计算指标
        rmse = np.sqrt(np.mean((y_pred - y_test_unscaled)**2))
        r2 = r2_score(y_test_unscaled, y_pred)
        print(f"\n测试集综合指标:")
        print(f"- RMSE: {rmse:.3f} dB")
        print(f"- R²分数: {r2:.3f}")
        
        # 各维度指标
        print("\n各维度表现:")
        for i in range(y_pred.shape[1]):
            col_rmse = np.sqrt(np.mean((y_pred[:,i] - y_test_unscaled[:,i])**2))
            col_r2 = r2_score(y_test_unscaled[:,i], y_pred[:,i])
            print(f"Q{i+1}: RMSE={col_rmse:.3f} dB | R²={col_r2:.3f}")
        
        # 预测可视化
        plot_predictions(y_test_unscaled, y_pred)
    # 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    torch.save(checkpoint, save_path)
    
    return model

# ----------------------
# 3. 自适应在线微调
# ----------------------
def adaptive_online_finetuning(model_path="enhanced_model.pth", 
                              num_samples=5,   # 建议至少5个样本
                              epochs=50,       # 增加训练轮次
                              lr=1e-4,        # 调整初始学习率
                              lip_lambda=0.2): # Lipschitz约束系数
    # ----------------------
    # 1. 模型加载与初始化
    # ----------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到预训练模型 {model_path}")

    # 初始化当前模型
    model = EnhancedLipschitzNN()
    
    try:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 加载模型参数（忽略scaler相关键值）
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 加载标准化器
        model.scaler_X = checkpoint['scaler_X']
        model.scaler_y = checkpoint['scaler_y']
        
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}") from e

    print("\n=== 成功加载增强模型 ===")

    # ----------------------
    # 2. 数据准备与增强
    # ----------------------
    # 加载原始数据集
    loaded_data = np.load("C:\\Users\\55266\\Desktop\\论文综述\\dataset_6wave_500sample_new.npz")
    setting_list = loaded_data['setting_list'].reshape(-1, 12)
    q_list = loaded_data['q_list'].reshape(-1, 6)
    
    # 数据增强函数
    def sliding_window_augmentation(data, window_size=3):
        """滑动窗口数据增强"""
        augmented = []
        for i in range(len(data)-window_size+1):
            window = data[i:i+window_size]
            # 添加均值扰动
            augmented.append(np.mean(window, axis=0))
            # 添加噪声扰动
            augmented.append(window[1] + np.random.normal(0, 0.01, window[1].shape))
        return np.array(augmented)
    
    # 获取原始数据并增强
    start_idx = random.randint(1, 50)  # 起始样本索引
    raw_configs = setting_list[start_idx:start_idx+num_samples*2]  # 原始数据
    raw_q = q_list[start_idx:start_idx+num_samples*2]
    
    # 应用数据增强
    aug_configs = sliding_window_augmentation(raw_configs)
    aug_q = sliding_window_augmentation(raw_q)

    # 合并数据集
    real_configs = np.vstack([raw_configs, aug_configs])
    real_q = np.vstack([raw_q, aug_q])

    # 标准化处理
    real_configs_scaled = model.scaler_X.transform(real_configs)
    real_q_scaled = model.scaler_y.transform(real_q)

    # 创建数据加载器
    dataset = TensorDataset(
        torch.FloatTensor(real_configs_scaled),
        torch.FloatTensor(real_q_scaled)
    )
    loader = DataLoader(dataset, 
                      batch_size=min(8, len(dataset)),  # 动态批次大小
                      shuffle=True,
                      pin_memory=True)

    # ----------------------
    # 3. 模型参数冻结策略
    # ----------------------
    # 冻结特征提取层
    for name, param in model.named_parameters():
        if "feature_extractor" in name:
            param.requires_grad = False
            print(f"冻结层: {name}")
    
    # 验证可训练参数
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"\n可训练参数: {trainable_params}")

    # ----------------------
    # 4. 优化器配置
    # ----------------------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    
    # 带热重启的余弦退火调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,        # 每10个epoch重启
        T_mult=1,
        eta_min=1e-6
    )
    
    # 早停机制
    best_loss = float('inf')
    patience = 8
    patience_counter = 0

    # ----------------------
    # 5. 训练循环
    # ----------------------
    print(f"\n=== 开始微调 (增强后样本数: {len(dataset)}) ===")
    losses = []
    grad_magnitudes = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_grad = 0.0
        
        for x, y in loader:
            x.requires_grad_(True)  # 启用梯度计算
            
            optimizer.zero_grad()
            
            # 前向传播
            pred = model(x)
            
            # 复合损失计算
            mse_loss = nn.MSELoss()(pred, y)
            lip_loss = model.lip_loss(x, pred)
            total_loss = mse_loss + lip_lambda * lip_loss
            
            # 反向传播
            total_loss.backward()
            
            # 梯度监控
           # 修正后的梯度裁剪（仅限可训练参数）
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=1.0,  #可降低阈值
                norm_type=2
            )
            
            # 记录梯度量级
            total_grad += grad_norm.item()
            
            optimizer.step()
            epoch_loss += total_loss.item()
        
        # 更新学习率
        scheduler.step()
        
        # 计算指标
        avg_loss = epoch_loss / len(loader)
        avg_grad = total_grad / len(loader)
        losses.append(avg_loss)
        grad_magnitudes.append(avg_grad)
        
        # 早停判断
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"fine_tuned_best.pth")
        else:
            patience_counter += 1
        
        # 打印训练信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"Loss: {avg_loss:.3e} | "
              f"Grad: {avg_grad:.2f} | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_counter}/{patience}")
        
        # 早停触发
        if patience_counter >= patience:
            print(f"早停触发，最佳loss: {best_loss:.4e}")
            break

    # ----------------------
    # 6. 训练可视化
    # ----------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, marker='o', color='#FF6B6B')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(grad_magnitudes, marker='^', color='#4ECDC4')
    plt.title("Gradient Magnitude")
    plt.xlabel("Epoch")
    plt.ylabel("Grad Norm")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # ----------------------
    # 7. 性能验证
    # ----------------------
    def create_baseline_model():
        """创建未微调的基准模型"""
        baseline_model = EnhancedLipschitzNN()
        baseline_model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'], 
                                     strict=False)
        baseline_model.scaler_X = model.scaler_X
        baseline_model.scaler_y = model.scaler_y
        baseline_model.eval()
        return baseline_model

    # 使用独立测试集（示例索引）
    test_indices = [15, 18, 22]  # 未参与训练的数据
    test_configs = setting_list[test_indices]
    test_q = q_list[test_indices]
    
    # 标准化处理
    test_configs_scaled = model.scaler_X.transform(test_configs)
    test_q_scaled = model.scaler_y.transform(test_q)
    
    # 预测对比
    model.eval()
    baseline_model = create_baseline_model()
    
    with torch.no_grad():
        # 加载最佳模型
        model.load_state_dict(torch.load("fine_tuned_best.pth"))
        
        # 获取预测结果
        baseline_pred = baseline_model(torch.FloatTensor(test_configs_scaled))
        finetuned_pred = model(torch.FloatTensor(test_configs_scaled))
        
        # 逆标准化
        def inverse_transform(y):
            return model.scaler_y.inverse_transform(y.numpy())
        
        # 计算指标
        def calculate_metrics(true, pred):
            rmse = np.sqrt(np.mean((true - pred)**2))
            mae = np.mean(np.abs(true - pred))
            return rmse, mae
        
        # 转换数据
        true_values = inverse_transform(torch.FloatTensor(test_q_scaled))
        baseline_preds = inverse_transform(baseline_pred)
        finetuned_preds = inverse_transform(finetuned_pred)
        
        # 计算整体指标
        baseline_rmse, baseline_mae = calculate_metrics(true_values, baseline_preds)
        finetuned_rmse, finetuned_mae = calculate_metrics(true_values, finetuned_preds)
        
        # 打印对比结果
        print("\n=== 独立测试集性能对比 ===")
        print(f"| 指标        | 基线模型    | 微调模型    | 改进率    |")
        print(f"|------------|------------|------------|----------|")
        print(f"| RMSE (dB)  | {baseline_rmse:.4f}   | {finetuned_rmse:.4f}   | {100*(baseline_rmse-finetuned_rmse)/baseline_rmse:.1f}%  |")
        print(f"| MAE (dB)   | {baseline_mae:.4f}   | {finetuned_mae:.4f}   | {100*(baseline_mae-finetuned_mae)/baseline_mae:.1f}%  |")
        
        # 可视化对比
        plt.figure(figsize=(15, 6))
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.plot(true_values[:, i], 'ko-', label='True')
            plt.plot(baseline_preds[:, i], 'r^--', label='Baseline')
            plt.plot(finetuned_preds[:, i], 'bs:', label='Fine-tuned')
            plt.title(f"Q{i+1} Prediction")
            plt.xlabel("Sample Index")
            plt.ylabel("Value")
            plt.legend()
        plt.tight_layout()
        plt.show()
     # 保存模型和标准化器的状态
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_X': model.scaler_X,
        'scaler_y': model.scaler_y
    }
    torch.save(checkpoint, f"fine_tuned_best.pth")

    return model

# ----------------------
# 4. 主控程序
# ----------------------
if __name__ == "__main__":
    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    set_seed()
    
    print("""
    ==============================
    增强型光网络优化系统
    ==============================
    运行模式选择：
    0 - 增强型离线训练
    1 - 自适应在线微调
    """)
    
    while True:
        choice = input("请输入模式选择 (0/1): ").strip()
        
        if choice == "0":
            enhanced_offline_training()
            break
        elif choice == "1":
            try:
                num_samples = int(input("请输入真实数据量 (1-5): ").strip())
                num_samples = max(1, min(5, num_samples))
            except:
                print("输入无效，默认使用1个样本")
                num_samples = 1
                
            adaptive_online_finetuning(
                num_samples=num_samples,
                epochs=50,      # 增加微调轮次
                lr=1e-4       # 优化学习率
            )
            break
        else:
            print("输入错误，请重新输入")