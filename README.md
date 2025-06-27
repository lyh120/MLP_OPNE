# MLP_OPNE
Some applications of MLP in the optical network.
## 🔧 Git 基础操作指南

### 1. 初始化仓库
```bash
# 创建新目录并初始化
mkdir my-project
cd my-project
git init

# 克隆现有仓库
git clone https://github.com/user/repo.git
```

### 2. 基本工作流
```bash
# 查看当前状态
git status

# 添加所有修改到暂存区
git add .

# 提交更改（添加描述信息）
git commit -m "添加新功能"

# 推送到远程仓库
git push origin main
```

### 3. 分支管理
```bash
# 创建新分支
git branch feature-x

# 切换到分支
git checkout feature-x
# 或（Git 2.23+）
git switch feature-x

# 合并分支到当前分支
git merge feature-x

# 删除已合并分支
git branch -d feature-x
```

### 4. 撤销操作
```bash
# 撤销未暂存的修改
git checkout -- filename

# 撤销暂存区的文件（保留工作区修改）
git reset HEAD filename

# 修改上次提交
git commit --amend
```

### 5. 远程仓库管理
```bash
# 添加新远程仓库
git remote add upstream https://github.com/other/repo.git

# 查看远程连接
git remote -v

# 拉取远程更新
git pull origin main

# 推送本地分支到远程
git push -u origin new-branch
```
#### 🔌 关联远程仓库补充
```bash
# 添加远程仓库地址（命名为origin，这是默认约定）
git remote add origin https://github.com/用户名/仓库名.git

# 验证是否添加成功
git remote -v
# 应该显示：
# origin  https://github.com/用户名/仓库名.git (fetch)
# origin  https://github.com/用户名/仓库名.git (push)

```
#### 🔄 首次同步（重要！）
```bash
# 如果远程仓库非空（已有README等文件），必须先拉取：
git pull origin main --allow-unrelated-histories
# ↑ 强制合并无关历史（首次必须）

# 如果远程仓库是空的（全新仓库）可跳过上一步
```
#### 扩展场景
重命名远程仓库（如将 new-origin 改为 origin）：
```bash
  git remote rename new-origin origin
```
修改 URL（如需更新 old-origin 的地址）：
```bash
  git remote set-url old-origin <新仓库URL>
```
一次性清空所有远程仓库：
```bash
  git remote remove new-origin
  git remote remove old-origin
  git remote remove origin  # 如果未被提前删除
```
注意：删除操作不可逆，确保不再需要该远程链接后再执行。如果后续需重新添加 origin，可使用：
> git remote add origin <仓库URL>

### 6. 查看历史记录
```bash
# 精简日志
git log --oneline

# 带分支图的日志
git log --graph --all --decorate

# 显示特定文件的修改历史
git log -p filename
```

### 7. 配置设置
```bash
# 设置全局用户名
git config --global user.name "Your Name"

# 设置全局邮箱
git config --global user.email "your@email.com"

# 查看所有配置
git config --list
```

> 💡 提示：使用 `--help` 查看详细帮助文档，例如：`git commit --help`

## 🔧 关于一些python编程的基础知识

### 1、pyhton中的self之面向对象的特点
C++ this指针是一个隐式指针，指向当前对象的地址（内存位置）。
编译器自动生成，无需显式声明。
类型为 ClassName* const（常量指针）
```c
class MyClass {
private:
    int value;
public:
    void setValue(int value) {
        this->value = value; // 用 this 区分成员变量和参数
    }

    MyClass* getAddress() {
        return this; // 返回当前对象地址
    }
};

// 使用
MyClass obj;
obj.setValue(42);//这时候的值，obj.value == 42了
MyClass* addr = obj.getAddress(); // 获取对象地址
```
Python self参数是方法的第一个显式参数，代表当前对象的引用。
必须手动在方法定义中声明（约定命名为 self）。
本质是一个普通参数，指向对象实例。
```python
class MyClass {
      def __init__(self,value):
          self.value = value #必须用self去访问成员变量

      def set_value(self,value):
          self.value = value #显式使用self

      def get_self(self):
          return self #返回当前对象的引用
}

#使用
obj = MyClass(42)
obj.set_value(10)
ref = obj.get_self() #ref和obj指向同一对象，引用同一个对象
```
底层机制:
C++
this 是编译器传递给成员函数的隐藏参数，函数调用时自动传入对象地址。
例如：obj.func(x) 被编译为 func(&obj, x)。
Python
self 是显式参数，调用方法时解释器自动将对象引用作为第一个参数传入。
例如：obj.method(x) 等价于 MyClass.method(obj, x)。

## 📦 代码版本管理
### Model版本管理
| 版本（model） | 主要功能 | 说明 | 日期 | 对应python文件 |
| --- | --- | --- | --- | --- |
| Basic_model | 利用LHS和高斯联合采样，基础MLP12*5结构，滑动窗口数据增强，实现了每隔4组数据进行一次在线微调| 最清晰易懂原始的版本 | 2025.06.27 | Basic_model.py |
| Basic_model_with_explanation | 利用LHS和高斯联合采样，基础MLP12*5结构，滑动窗口数据增强，实现了每隔4组数据进行一次在线微调| 加上了海量注释的原始版本 | 2025.06.27 |Basic_model_with_explanation.py |
| Updated_model | 利用多策略协同全局搜索进行采点，并带有屏幕快照与峰值回退功能，实现了每隔4组数据进行一次在线微调| 借助AI编写的较为复杂的代码。其中实验结果证明峰值回退时当出现一次的猛烈下降很大程度上证明模型预测已经趋于最高了 | 2025.06.27 | Updated_model.py |
### 其他文件版本管理
| 对应的python文件名 | 说明 |
| --- | --- |
| Prediction_Network.py | 将特征提取层与预测头相互分离的神经网络结构，离线测试版本，在线微调的基础代码|
| Offline_train.py | 离线训练代码，训练模型并保存，做到提前热身工作，供后续在线微调代码加载提前训练好的模型与使用|

