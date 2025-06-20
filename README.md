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
