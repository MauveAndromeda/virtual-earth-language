# 🌍 Virtual Earth: Emergent Language Evolution

> Watch AI agents spontaneously create, evolve, and standardize their own languages across virtual continents.

**Ubuntu优化版本** - 支持CUDA加速和多进程处理

## 🚀 Ubuntu快速启动

### 一键设置（推荐）
```bash
# 克隆或下载项目
git clone https://github.com/yourusername/virtual-earth-language.git
cd virtual-earth-language

# 运行Ubuntu设置脚本
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh

# 快速启动
bash scripts/quick_start_ubuntu.sh
```

### 手动设置
```bash
# 1. 安装系统依赖
sudo apt update
sudo apt install -y python3-dev build-essential curl wget git

# 2. 创建conda环境
conda env create -f environment.yml
conda activate virtual-earth

# 3. 安装项目
pip install -e .

# 4. 运行测试
pytest tests/test_setup.py

# 5. 开始实验
python experiments/main_experiments.py
```

## 🎯 Ubuntu特性

- ✅ **GPU自动检测** - CUDA自动配置和优化
- ✅ **多进程支持** - 利用所有CPU核心
- ✅ **内存优化** - 适配Ubuntu系统特性
- ✅ **容器就绪** - Docker和Singularity支持
- ✅ **HPC友好** - SLURM作业调度兼容

## 📊 系统要求

- **OS**: Ubuntu 18.04+ (推荐 20.04/22.04)
- **Python**: 3.8+
- **RAM**: 8GB+ (推荐 16GB+)
- **GPU**: NVIDIA GPU (可选，自动检测CUDA)
- **Storage**: 10GB+ 可用空间

## 🔧 开发工具

```bash
# 代码格式化
black src/ tests/ experiments/

# 类型检查
mypy src/

# 运行所有测试
pytest tests/ --cov=src

# 启动Jupyter
jupyter notebook notebooks/

# 监控GPU
watch nvidia-smi
```

## 📁 项目结构

```
virtual-earth-language/
├── src/                    # 核心实现
│   ├── envs/              # 环境模块
│   ├── agents/            # 智能体
│   └── ...
├── experiments/           # 实验脚本  
├── configs/              # 配置文件
├── data/                 # 数据集
├── scripts/              # Ubuntu脚本
└── tests/                # 测试代码
```

## 🚀 使用示例

### 基础实验
```bash
python experiments/main_experiments.py
```

### 地理实验
```bash
python experiments/main_experiments.py --config-name geography/mountains
python experiments/main_experiments.py --config-name geography/islands
```

### 批量实验
```bash
bash scripts/run_experiments.sh
```

### GPU加速
```bash
python experiments/main_experiments.py training.device=cuda
```

## 📈 实验监控

```bash
# TensorBoard
tensorboard --logdir outputs/

# 实时监控
htop
nvidia-smi -l 1
```

## 🐛 故障排除

### CUDA问题
```bash
# 检查CUDA安装
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 重装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 环境问题
```bash
# 重建环境
conda deactivate
conda env remove -n virtual-earth
conda env create -f environment.yml
```

## 📚 更多文档

- 📖 [完整文档](docs/)
- 🧪 [实验指南](docs/experiments.md)
- 🔧 [开发指南](docs/development.md)
- 🐳 [Docker部署](web/docker/)

---

**为Ubuntu优化，在Linux上获得最佳性能！** 🐧
