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


