#!/bin/bash
# Ubuntu快速启动脚本

set -e

echo "🌍 Virtual Earth - Ubuntu快速启动"

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装，请先运行主设置脚本"
    exit 1
fi

# 激活环境
echo "🐍 激活环境..."
eval "$(conda shell.bash hook)"

# 创建环境（如果不存在）
if ! conda env list | grep -q virtual-earth; then
    echo "📦 创建conda环境..."
    conda env create -f environment.yml
fi

conda activate virtual-earth

# 安装项目
echo "🔧 安装项目..."
pip install -e .

# 运行测试
echo "🧪 运行基础测试..."
python -m pytest tests/test_setup.py -v

# 运行演示实验
echo "🚀 运行演示实验..."
python experiments/main_experiments.py

echo "✅ 快速启动完成!"
echo "📚 使用方法:"
echo "  conda activate virtual-earth"
echo "  python experiments/main_experiments.py"
echo "  bash scripts/run_experiments.sh"
