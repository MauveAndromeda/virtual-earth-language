#!/bin/bash
# Ubuntu优化的批量实验脚本

echo "🧪 Virtual Earth Language Evolution - Ubuntu实验脚本"
echo "🐧 检测系统信息..."

# 检测系统资源
CPU_CORES=$(nproc)
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
TOTAL_MEM=$(free -h | awk '/^Mem/ {print $2}')

echo "💻 CPU核心数: $CPU_CORES"
echo "🚀 GPU数量: $GPU_COUNT"
echo "💾 总内存: $TOTAL_MEM"

# 检测CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU检测到:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "ℹ️ 未检测到NVIDIA GPU，将使用CPU"
fi

# 激活conda环境（如果存在）
if command -v conda &> /dev/null; then
    echo "🐍 激活conda环境..."
    eval "$(conda shell.bash hook)"
    conda activate virtual-earth 2>/dev/null || echo "⚠️ virtual-earth环境未找到"
fi

echo ""
echo "🧪 开始运行实验..."

# 基础实验
echo "📊 运行基础实验..."
python experiments/main_experiments.py \
    hydra.job.name=basic_experiment \
    training.device=auto

# 地理变体实验
if [ -f "configs/geography/mountains.yaml" ]; then
    echo "🏔️ 运行山地实验..."
    python experiments/main_experiments.py \
        --config-path configs/geography \
        --config-name mountains \
        hydra.job.name=mountains_experiment
fi

if [ -f "configs/geography/islands.yaml" ]; then
    echo "🏝️ 运行岛屿实验..."
    python experiments/main_experiments.py \
        --config-path configs/geography \
        --config-name islands \
        hydra.job.name=islands_experiment
fi

echo "✅ 所有实验完成!"
echo "📁 结果保存在 outputs/ 目录中"

# 显示结果摘要
echo ""
echo "📊 实验结果摘要:"
find outputs -name "*.yaml" -type f | head -5 | while read file; do
    echo "  - $file"
done
