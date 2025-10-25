"""
FJSP-AGV 集成调度研究项目
考虑AGV电池约束的柔性作业车间绿色集成调度多目标优化研究
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 导入主要模块以便直接访问
from .models import FJSPAGVModel, ModelValidator
from .utils import DataGenerator
from .config import ModelConfig, AlgorithmConfig

__all__ = [
    'FJSPAGVModel',
    'ModelValidator',
    'DataGenerator',
    'ModelConfig',
    'AlgorithmConfig'
]