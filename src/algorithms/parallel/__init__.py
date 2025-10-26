"""
并行计算模块
"""

from .parallel_manager import ParallelManager, ParallelConfig, ThreadResult
from .solution_rotator import SolutionRotator

__all__ = ['ParallelManager', 'ParallelConfig', 'ThreadResult', 'SolutionRotator']