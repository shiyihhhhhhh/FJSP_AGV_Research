# FJSP-AGV 集成调度研究

考虑AGV电池约束的柔性作业车间绿色集成调度多目标优化研究

## 项目结构
- `src/models/`: 数学模型实现
- `src/algorithms/`: 优化算法实现  
- `src/utils/`: 工具函数
- `src/config/`: 配置管理
- `src/experiments/`: 实验脚本
- `data/`: 数据目录
- `notebooks/`: Jupyter notebooks

## 快速开始
1. 安装依赖: `pip install -r requirements.txt`
2. 运行模型验证: `python main.py --mode model`
3. 运行算法测试: `python main.py --mode algorithm`

## 开发阶段
- ✅ 阶段1: 数学模型建立与验证
- 🔄 阶段2: 混合智能算法设计
- ⏳ 阶段3: 实验分析与论文撰写