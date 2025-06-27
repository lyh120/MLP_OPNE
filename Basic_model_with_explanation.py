import torch
import os
import time
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import qmc
from torch.utils.data import DataLoader,TensorDataset
from utilities.utils import apply_parallel, query_parallel, query_setting

class LipschitzNN(nn.Module):   #表示自定义的类继承了pytorch提供的基类
    def __init__(self,L_lambda,L_const,input_dim=12,output_dim=5):
        super().__init__() 
        #LipschitzNN 类继承自 nn.Module（PyTorch的神经网络基类）
        #super().__init__() 会调用 nn.Module 的 __init__() 方法
        
        self.Net = nn.Sequential( #相当于一个“层的流水线”，简化了网络结构的定义
            #在调用forward（）时，会自动按顺序执行所有层
            nn.Linear(input_dim,32), #第1层，线性层，将输入维度为12的向量映射到32维
            nn.GELU(), #第2层，激活函数层，使用GELU函数
            # 比ReLU更平滑，在负值区域也有微小梯度，训练更稳定
            nn.Linear(32,16), #第3层，线性层，将32维的向量映射到16维
            nn.GELU(), #第4层，激活函数层，使用GELU函数
            nn.Linear(16,output_dim), #第5层，线性层，将16维的向量映射到输出维度5维
        )
        #为什么要有激活函数？ 
        # 为了引入非线性因素，使得神经网络能够学习复杂的函数映射
        # 若没有激活函数，无论神经网络有多少层，都是线性组合，无法表达任意函数
        # 加入激活函数后，神经网络可以逼近任意函数，逼近能力更强
        # 神经网络本质上是一种优化问题，优化逼近目标函数
        # 而如果不引入非线性因素，优化问题退化为线性规划，无法逼近复杂函数
        # 所以，神经网络的非线性结构是由激活函数引入的
        self.lip_lambda = L_lambda
        self.scaler_X = None
        self.scaler_y = None
        self.Lip_const = torch.FloatTensor(L_const) #将输入的 L_const 转换为 PyTorch 的浮点张量 (32-bit float tensor)。
    
    def forward(self,x):
        return self.Net(x)
    
    def lip_loss(self,x,y_pred):
        sigma_x = torch.FloatTensor(self.scaler_X.scale_)
        # 计算输入数据的标准差
        #self代指当前类的实例对象，即LipschitzNN这个继承了nn.model的自定义类
        #self.scaler_X.scale_ 是一个 NumPy 数组，包含每个特征的标准差
        #将其转换为 PyTorch 张量，以便在 PyTorch 模型中使用
        # 每个特征的标准差不同，代表了不同的动态范围，需要对其进行归一化
        # 归一化后，每个特征的取值范围都是[-1,1]，可以避免某个特征对损失函数的贡献过大
        # 从而使模型更加稳定，避免过拟合
        sigma_y = torch.FloatTensor(self.scaler_y.scale_)
        # 计算输出数据的标准差
        #self.scaler_y.scale_ 是一个 NumPy 数组，包含每个输出的标准差
        #将其转换为 PyTorch 张量，以便在 PyTorch 模型中使用
        # 每个输出的标准差不同，代表了不同的动态范围，需要对其进行归一化
        Lip_scaled = self.Lip_const*(sigma_x.reshape(1,-1)/sigma_y.reshape(1,-1))
        # 计算 lipschitz 矩阵的标准化
        batch_size = x.size(0)
        # x是输入张量，形状为[batch_size, input_dim]
        # x.size(0)获取当前批次的样本数量
        # 为后续计算雅可比矩阵准备批次维度信息

        # 初始化雅可比矩阵
        # 雅可比矩阵是一个矩阵，包含了函数在每个点的局部变化率
        # 对于神经网络，雅可比矩阵描述了输入空间中每个点的输出对输入的敏感度
        # 雅可比矩阵的元素是函数在该点的局部变化率，代表了函数在该点的变化趋势
        # 雅可比矩阵的元素越大，代表函数在该点的变化越快，即函数在该点的局部变化率越大
        jacobian = torch.zeros(batch_size,y_pred.size(1),x.size(1)).to(x.device)
        #通过.to(x.device)确保该矩阵与输入数据在同一设备（CPU/GPU）上
        for i in range(y_pred.size(1)):
            gradients = torch.autograd.grad(
                outputs=y_pred[:,i].sum(),
                inputs=x,
                create_graph=True, #保留计算图以便二阶导计算
                retain_graph=True  #保留计算图供后续循环使用
            )[0]
            
            # 计算每个输出维度的梯度
            # y_pred[:,i].sum() 表示对第i个输出维度求和
            # x 表示输入数据
            # create_graph=True 表示创建计算图，以便后续计算更高阶导数
            # 得到的 gradients 是一个张量，形状为[batch_size, input_dim]
            # 表示每个样本在第i个输出维度上的梯度
            jacobian[:,i,:] = gradients 

            jacobian_abs = torch. abs(jacobian)
            Lip_expanded = Lip_scaled.unsqueeze(0).expand(batch_size,-1,-1)  
            #Lip_scaled.unsqueeze(0)：
            # 在维度0（最外层）添加一个大小为1的维度
            # 例如：原形状为[5,12] → 变为[1,5,12]
            # .expand(batch_size, -1, -1)：
            # 将第0维从1扩展到batch_size（复制batch_size份）
            # -1表示保持该维度大小不变
            # 最终形状变为[batch_size,5,12]
            excess = (jacobian_abs - Lip_expanded).clamp(min=0)
            # 计算每个元素的残差（即大于0的部分）
            # 残差代表了当前样本在该输出维度上的梯度超出了 lipschitz 矩阵的敏感度范围
            # 残差越大，说明当前样本在该输出维度上的梯度变化越快，越超出敏感度范围
            # 残差越小，说明当前样本在该输出维度上的梯度变化越慢，越在敏感度范围内
            # 残差的存在可能导致模型不稳定，需要对其进行惩罚
            #只保留那些违反Lipschitz约束的部分（即雅可比矩阵元素大于Lipschitz约束值的部分）
            # 忽略那些满足约束的部分（将它们置为0）
            # 最终得到的excess张量只包含违反约束的程度信息
            return torch.mean(excess**2)
            """这个返回值衡量了网络在整个batch上违反Lipschitz约束的平均程度
               值越大表示违反约束越严重
               在训练时会作为正则项被最小化，迫使网络满足Lipschitz约束
               loss = 1/N Σ_{i,j} [max(0, |∂y_i/∂x_j| - L_{i,j})]²
               其中N是batch中所有元素的数量（batch_size × 输出维度 × 输入维度）
               这个损失函数的设计确保了：
               只有当网络违反约束时才会产生惩罚
               违反程度越大惩罚越强
                最终使网络的局部灵敏度(Lipschitz常数)保持在预定范围内
            """

def resolution_shape(max_gradient, settings, qs):
    """生成一个约束函数，用于评估目标配置是否满足质量要求
    
    参数:
        max_gradient: 最大梯度矩阵，形状为[5,12]，表示每个Q指标对每个参数的敏感度
        settings: 历史配置列表，每个配置是12维向量
        qs: 对应的历史Q值列表，每个Q值是5维向量
        
    返回:
        一个约束函数，输入目标配置x_tar，返回是否满足所有Q值要求(1=满足,0=不满足)
    """
    mg = np.array(max_gradient)  # 转换为numpy数组
    
    def constraint(x_tar):
        """约束函数实现
        
        参数:
            x_tar: 目标配置，12维向量
            
        返回:
            1: 如果目标配置在所有历史数据评估下都满足Q值>9
            0: 其他情况
        """
        for set, q in zip(settings, qs):  # 遍历所有历史配置和Q值
            delta = abs(np.array(set) - x_tar)  # 计算参数变化量
            degrade = np.dot(mg, delta)  # 计算Q值下降量 = 敏感度矩阵·参数变化量
            q_pred = q - degrade  # 预测新配置下的Q值
            if min(q_pred) > 9:  # 如果所有Q值预测都>9
                return 1  # 满足约束
        return 0  # 不满足约束
    
    return constraint  # 返回约束函数 

def generate_best_solution(settings,qs,cons,bounds,model,
                           batch_size=100,noise_scale=0.05,noise_increment=0.1,
                           target_points=500,max_iters=100):
        
    """混合式最优解生成函数"""
    def _scale_params(x, bounds):
        """将输入参数归一化到[0,1]区间
        
        Args:
            x: 待归一化的参数数组，形状为(..., n_dims)
            bounds: 各参数的取值范围列表，每个元素为(lower_bound, upper_bound)元组
            
        Returns:
            归一化后的参数数组，各维度值在[0,1]之间
        """
        x = np.array(x)  # 确保输入为numpy数组
        x_norm = np.zeros_like(x, dtype=float)  # 初始化输出数组
        for i, (lb, ub) in enumerate(bounds): #enumerate(bounds) 会生成 (index, value)
            # 对每个维度进行线性归一化：(x - lb)/(ub - lb)
            x_norm[..., i] = (x[..., i] - lb) / (ub - lb)
        return x_norm
    
    def _model_predict(model, inputs):
        """使用模型进行预测并返回numpy数组结果
        Args:
            model: 训练好的神经网络模型
            inputs: 输入数据(可以是numpy数组或类似数组结构)
        Returns:
            numpy数组形式的预测结果(各输出维度求和后的结果)
        """
        with torch.no_grad(): # 禁用梯度计算，提高预测效率
            inputs_tensor = torch.FloatTensor(inputs)  # 将输入转换为PyTorch张量
            preds = model(inputs_tensor).sum(dim=1)  # 模型预测并对输出维度求和
            return preds.numpy()  # 将结果转换回numpy数组
    #获取参数边界信息
    lower_bounds = np.array([b[0] for b in bounds]) #获取所有参数的下界
    upper_bounds = np.array([b[1] for b in bounds]) #获取所有参数的上界
    dim = len(bounds)  # 确定参数维度

    #混合采样参数
    lhs_radio = 0.7 #拉丁超立方采样比例
    current_point = settings[-1] #获取当前解即最优解

    satisfied_points = [] #存放满足约束条件的候选解
    
    #主循环：生成候选解
    for _ in range(max_iters):
        all_points = [] #存放当前采样获得的所有采样点
        
        #围绕当前最优解生成局部搜索范围
        center = current_point
        #计算局部搜索下界，确保不超出全局边界
        local_lb = np.clip(center - noise_scale*(upper_bounds-lower_bounds),lower_bounds,upper_bounds)
        #计算局部搜索上界，确保不超出全局边界
        local_ub = np.clip(center + noise_scale*(upper_bounds-lower_bounds),lower_bounds,upper_bounds)
        
        #拉丁超立方采样（LHS）- 在局部范围内生成均匀分布的样本
        """
        第一行：
        qmc.LatinHypercube(d=dim)：创建一个拉丁超立方采样器，dim是参数空间的维度（这里是12维）
        .random(n=int(batch_size*lhs_ratio))：生成指定数量的样本点，batch_size是总采样数，lhs_ratio=0.7表示LHS采样占70%
        生成的样本在[0,1]^dim的单位超立方体内均匀分布，且每个维度的投影都是均匀分布的
        """
        """
        第二行：
        qmc.scale()：将单位超立方体中的样本映射到实际的参数范围
        local_lb和local_ub：定义了局部搜索范围的下界和上界
        最终得到的lhs_samples是在局部搜索范围内均匀分布的样本点
        """

        lhs_samples = qmc.LatinHypercube(d=dim).random(n=int(batch_size*lhs_radio))
        lhs_samples = qmc.scale(lhs_samples,local_lb,local_ub)

        #高斯采样 - 在最优解周围生成随机扰动的样本
        gauss_samples = center + np.random.normal(
            scale=noise_scale,#扰动的幅度
            size=(int(batch_size*(1-lhs_radio)),dim)#采样数量和维度
            )*(upper_bounds-lower_bounds)#参数范围缩放
        #合并两种采样结果后再处理
        candidates = np.vstack([lhs_samples,gauss_samples]) 
        condidatas = np.clip(candidates,lower_bounds,upper_bounds)
        condidatas = np.round(condidatas,1)
        all_points.extend(candidates)
        
        #确保当前最优解包含在候选集里面
        all_points.extend(current_point)
        all_points = np.unique(all_points,axis=0) #删除重复行
        
        #筛选满足约束条件的候选点
        new_valid = [pt for pt in all_points if cons(pt)==1]
        satisfied_points.extend(new_valid)

        #如果找到足够多的有效候选点，提前终止
        if len(satisfied_points)>= target_points:
            break
    
    if not satisfied_points:
       print("没有找到满足约束的有效解，返回当前值")
       return current_point
    
    #使用模型评估候选解的质量
    candidates_scaled = _scale_params(satisfied_points,bounds)
    scores = _model_predict(model,candidates_scaled)
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_solution = satisfied_points[best_idx]
    
    #比较找到的解是否优于当前最优解
    current_score = _model_predict(model,_scale_params([current_point],bounds))[0]
    if best_score > current_score:
        print(f"新解的分数：{best_score}")
        print(f"当前轮次最优解的参数：{best_solution}")
        print(f"找到更优解：{best_score:.2f}>{current_score:.2f}")
        return best_solution
    else:
        #找不到比当前解更好的解，在联合搜索策略里面可以通过适当增大搜索范围与搜索点数
        new_noise = noise_scale*(1 + noise_increment)
        new_target_points = target_points*(1 + noise_increment)
        print(f"没有找到更好的解，当前最优分数：{current_score}")
        print(f"未找到改进，扩大噪声至：{new_noise:.3f},扩大采点数目到：{new_target_points}")
        return generate_best_solution(
            settings=settings,#当前最优解已经包含在settings和qs中，所以不需要vstack进行添加
            qs=qs,
            cons=cons,
            bounds=bounds,
            model=model,
            batch_size=batch_size,
            noise_scale=new_noise,
            noise_increment=noise_increment,
            target_points=new_target_points,
            max_iters=max_iters
        )
        
def adaptive_online_finetuning(
        model=None,num_samples=4,epochs=50,lr=1e-4,
        Lip_lambda=0.2,L_const=np.ones([5,12]),
        setting_list=None,q_list=None,save_path='enhanced_model.pth'
):  
    """"在线微调函数""" 
    #整合为二维数组
    setting_list = setting_list.reshape(-1,12)
    q_list = q_list.reshape(-1,5)

    #条件判断，提高安全性：数据不足时直接返回原来的模型
    if setting_list.shape[0]<num_samples:
        print("警告：提供的数据样本数量不足4组，无法进行在线微调")
        return model
    
    #滑动窗口数据增强
    def sliding_window_augmentation(data, window_size=3):
        """滑动窗口数据增强函数
        参数:
            data: 输入数据数组，形状为(n_samples, n_features)
            window_size: 滑动窗口大小，默认为3
        返回:
            增强后的数据数组，形状为(2*(n_samples-window_size+1), n_features)
        """
        augmented = []  # 初始化增强数据列表
        
        # 滑动窗口遍历数据
        for i in range(len(data)-window_size+1):
            window = data[i:i+window_size]  # 获取当前窗口数据
            
            # 添加窗口均值作为增强数据
            augmented.append(np.mean(window, axis=0))
            
            # 添加窗口中间点加噪声作为增强数据
            augmented.append(window[1] + np.random.normal(0, 0.01, window[1].shape))
            
        return np.array(augmented)  # 转换为numpy数组返回
    # 不断使用最近，最新加入的4组数据进行增强
    raw_configs = setting_list[-4:]
    raw_q = q_list[-4:]
    
    aug_configs = sliding_window_augmentation(raw_configs)
    aug_q = sliding_window_augmentation(raw_q)
     
   
    real_configs = np.vstack([raw_configs, aug_configs])
    real_q = np.vstack([raw_q, aug_q])
    """
    从在线学习的特性出发，只用最新4组增强后的数据。
    1、时效性优先：在线学习场景下，系统状态可能快速变化，最新数据最能反映当前状态
    2、避免概念漂移：历史增强数据可能包含过时模式，与新数据分布不一致
    """
    #标准化处理
    """
    scaler_X 是在模型预训练时初始化的 StandardScaler 对象
    它保存了原始训练数据的统计特性(均值和标准差)
    transform() 方法使用这些统计量对新数据进行标准化
    """
    real_configs_scaled = model.scaler_X.transform(real_configs)
    real_q_scaled = model.scaler_Y.transform(real_q)
    
    #创建数据集与数据加载器
    dataset = TensorDataset(
        torch.tensor(real_configs_scaled)
        ,torch.tensor(real_q_scaled)
    )
    loader = DataLoader(dataset,  # 要加载的数据集对象
                  batch_size=min(8, len(dataset)),  # 每批数据的大小
                  shuffle=True,  # 是否打乱数据顺序
                  pin_memory=True)  # 是否将数据固定在内存中
    """
    shuffle=True:
    在每个epoch开始时打乱数据顺序,防止模型学习到数据顺序带来的偏差,提高训练效果
    pin_memory=True:
    将数据固定在内存中,当使用GPU时，可以加速CPU到GPU的数据传输,特别适合小批量数据的情况
    """ 

    #定义优化器+余弦退火带重启的学习率调度策略
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad,model.parameters()),#确保只优化那些需要梯度更新的参数
        lr=lr,#初始化学习率为1e-4
        weight_decay = 1e-4 #权重衰减（L2正则化）系数，防止过拟合
    )

    #定义余弦退火带重启的学习率调度策略
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10, #初始周期长度
        T_mult=1, #每个周期结束后，周期长度的倍增倍数(1表示周期长度不变)
        eta_min=1e-6 #学习率下限
    )

    best_loss = float('inf')
    patience = 8
    patience_counter = 0
    losses = []
    grad_magnitudes = []

    print(f"\n=== 开始微调 (增强后样本数：{len(dataset)}) ===") #由于每次都是取最新的4组数据，所以正常情况增强后数据一定为8组
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_grad = 0.0
        
        for x,y in loader: #遍历数据加载器
            x.requires_grad_(True) #启用输入梯度计算
            optimizer.zero_grad() #清空之前的梯度
            
            pred = model(x) #前向传播,进行模型的预测
            mse_loss = nn.MSELoss()(pred,y) #计算MSE损失
            lip_loss = model.lip_loss(x,pred) #计算Lipschitz损失
            total_loss = mse_loss + model.lip_lambda*lip_loss#组合总损失 
            
            total_loss.backward() #反向传播计算梯度
            
            #梯度裁剪防止梯度爆炸
            grad_norm= torch.nn.utils.clip_grad_norm_(
                model.parameters(),#要裁剪模型参数
                max_norm=1.0, #对模型参数的梯度进行裁剪,确保范数不超过1.0
                norm_type=2 #使用范数类型：L2范数，
            )
            """
            L2范数（欧几里得范数）
            数学定义：

            对于向量x = [x₁, x₂, ..., xn]，其L2范数为：||x||₂ = √(x₁² + x₂² + ... + xn²)
            在梯度裁剪中，计算的是所有参数梯度的L2范数
            """
            total_grad += grad_norm.item()#计算整个epoch中梯度的平均大小（在后续代码中会用到avg_grad = total_grad / len(loader)），用于监控训练过程中梯度的大小变化情况。
            epoch_loss += total_loss.item()#累计损失
            optimizer.step() #根据计算的梯度更新参数
            
        scheduler.step() #更新学习率

        #计算当前epoch的平均损失和梯度范数
        avg_loss = epoch_loss/len(loader)
        avg_grad = total_grad/len(loader)
        losses.append(avg_loss)
        grad_magnitudes.append(avg_grad)
        
        #早停机制，当验证损失不再下降时，提前终止训练
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0 #重置耐心计数器
            torch.save(model.state_dict(),"fine_tuned_best.pth")#保存局部最佳模型，后续并未使用，处于保险起见进行过程保存
            """
            model.state_dict()：
            获取模型的所有可学习参数(权重和偏置)
            返回一个有序字典(OrderedDict)，包含各层的参数名和对应的张量值
            不包含模型结构信息，只保存参数值
            """    
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']#获取当前学习率的值
        print(f"Epoch {epoch+1:03d}/{epochs} | "
        f"Loss: {avg_loss:.3e} | "
        f"Grad: {avg_grad:.2f} | "
        f"LR: {current_lr:.2e} | "
        f"Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"早停触发，最佳loss：{best_loss:.4e}")
            break
    
    checkpoint = {
        'model_state_dict': model.state_dict(),#保存模型参数
        'scaler_X': model.scaler_X,# 保存输入数据的标准化器
        'scaler_y': model.scaler_y#  保存输出数据的标准化器
    }
    torch.save(checkpoint,save_path)#这里应该还是保存到我指定的save_path里面，确保每一次的微调都会被记录
    return model

def online_opt(x0,iter_num=20,seed=42,early_stop=True,patience=5,
               Lip_lamda=0.2, max_gradient=np.ones([5,12]),save_name='Basic_model',
               model_path='enhanced_model.pth'
):
    """
    先检查模型文件是否存在，不存在就抛出异常终止程序；存在则创建新的神经网络实例。这是一种防御性编程的做法，确保程序不会在缺少必要文件的情况下继续执行。
    """
    if not os.path.exist(model_path):
        raise FileNotFoundError(f"找不到预训练模型 {model_path} ")
    model = LipschitzNN(L_lambda=Lip_lamda,L_const=max_gradient)
    try:
        # 尝试加载模型检查点文件到CPU设备
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 加载模型参数（允许部分参数不匹配，strict=False）
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 加载数据标准化器
        model.scaler_X = checkpoint['scaler_X']
        model.scaler_y = checkpoint['scaler_y']
    
    except Exception as e:
        # 如果出现任何异常，抛出新的运行时错误并包含原始错误信息
        raise RuntimeError(f"模型加载失败: {str(e)}") from e

    print("\n=== 成功加载增强模型 ===")
    bounds = [(14, 23), (22, 30), (16, 22), (8, 15), (22,30), 
             (16, 22),(-2, 0), (-2, 0), (-2, 0), (-2, 0), (-2, 0), (-2, 0)]
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    
    random.seed(seed)
    np.random.seed(seed)
    """
    random.seed(seed)
    设置Python内置random模块的随机数生成器种子
    影响所有使用random模块生成的随机数序列
    
    np.random.seed(seed)
    设置NumPy库的随机数生成器种子
    影响所有使用np.random生成的随机数序列
    """
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
    #开始实际场景中参数的部署与调试
    start = time.time()
    
    for index in range(iter_num):
        cons = resolution_shape(L_const,settings,qs)#请注意最后返回的是一个约束函数cons
        new_solution = generate_best_solution(settings,qs,cons,bounds,model) #产生新解
        apply_parallel(new_solution)
        current_Q = query_parallel(avg_time=3)
       
        """
        np.concatenate():

        将原有的settings数组和新解在垂直方向(axis=0)拼接
        相当于在settings数组的最后添加一行新数据
        """
        settings = np.concatenate((settings,new_solution.reshape(1,-1)),axis=0)
        qs = np.concatenate((qs,current_Q.reshape(1,-1)),axis=0)

        #每积累4组数据才进行微调
        if (index + 1) % 4 ==0:
            model = adaptive_online_finetuning(
                model=model,
                setting_list=settings,
                q_list=qs,
                save_path='enhanced_model.pth'
            )
        #获得当前的Q值
        current_fitness_value = np.mean(current_Q)
        print(f"Iteration {index}:Current Fitness = {current_fitness_value}")
        
        if current_fitness_value > best_fitness:
            best_fitness = current_fitness_value
            best_solution = new_solution
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        fitness_history.append(current_fitness_value)
        q_min_history.append(np.min(current_Q))
        q_avg_history.append(np.mean(current_Q))

        np.savez(save_name+'.npz',qs=np.array(qs), settings=np.array(settings), 
                best_setting=np.array(best_solution), best_q=np.squeeze(best_Q))
        # 保存当前迭代的模型参数
        torch.save(model.state_dict(), f'{save_name}_iter_{index}.pth')
        
        if early_stop and (no_improvement_count>=patience):
            print(f"Early stoppping at Iteration {index}")
            break
    #可视化数据部分
    plt.figure(figsize=(12,6)) #创建一个新的图形窗口并指定图形大小,长12英寸，宽6英寸
    plt.subplot(1,2,1)
    plt.plot(fitness_history,label='Average Fitness') #label='Average Fitness' 为该折线图设置标签，用于在图例中显示。
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Fitness over Iterations')
    plt.legend() #显示图例

    plt.subplot(1,2,2)
    plt.plot(q_min_history, label='Minimum Q Value')
    plt.plot(q_avg_history, label='Average Q Value')
    plt.xlabel('Iteration')
    plt.ylabel('Q Value')
    plt.title('Q Value over Iterations')
    plt.legend() #显示图例

    plt.tight_layout()#plt.tight_layout() 用于自动调整子图参数，使子图之间的间距和边距更加合适，避免标签重叠。
    plt.show()#plt.show() 用于显示绘制好的图形。
    plt.savefig('Basic_model.png') #保存可视化结果
    
    


    end = time.time
    print(f"Optimal Solution:{best_solution}")
    print(f"Best Fitness:{best_fitness},Q Value:{best_Q}")
    print(f"Total Time:{end - start:.2f}s")


if __name__=='__main__':    
    Lip_data = np.load('./Lip_GN_5wave_0319.npz')
    Lip = Lip_data['mgn_list'][-1]
    #x = [16,25,18,10,25,18,-1,-1,-1,-1,-1,-1]
    x = [18.8 ,29. , 18.7, 14.6 ,27.7 ,21.7 ,-0.1, -0.7 ,-0.4 ,-0.7 ,-1.2 ,-1.7]
    online_opt(x0=x,iter_num=20,seed=42,early_stop=True,patience=5,
               Lip_lamda=0.2, max_gradient=Lip,model_path='enhanced_model.pth',
               save_name='Basic_model_online')