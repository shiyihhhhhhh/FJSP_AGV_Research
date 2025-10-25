# FJSP-AGV 研究项目开发说明

## 项目结构
FJSP_AGV_Research/
├── src/ # 源代码
│ ├── models/ # 数学模型
│ ├── algorithms/ # 优化算法
│ ├── utils/ # 工具函数
│ ├── config/ # 配置管理
│ └── experiments/ # 实验脚本
├── tests/ # 单元测试
├── data/ # 数据文件
├── notebooks/ # Jupyter notebooks
└── results/ # 实验结果


## 开发流程

### 1. 环境配置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt