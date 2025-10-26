"""
算法基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAlgorithm(ABC):
    """算法基类"""

    def __init__(self, name: str = "BaseAlgorithm"):
        self.name = name
        self.history = {
            'iterations': [],
            'best_fitness': [],
            'execution_time': []
        }

    @abstractmethod
    def initialize(self, problem_data: Dict[str, Any]):
        """初始化算法"""
        pass

    @abstractmethod
    def run(self, max_iterations: int) -> Dict[str, Any]:
        """运行算法"""
        pass

    @abstractmethod
    def get_best_solution(self) -> Any:
        """获取最优解"""
        pass

    def log_iteration(self, iteration: int, best_fitness: float, time_elapsed: float):
        """记录迭代信息"""
        self.history['iterations'].append(iteration)
        self.history['best_fitness'].append(best_fitness)
        self.history['execution_time'].append(time_elapsed)

    def get_statistics(self) -> Dict[str, Any]:
        """获取算法统计信息"""
        if not self.history['best_fitness']:
            return {}

        return {
            'best_fitness': min(self.history['best_fitness']),
            'worst_fitness': max(self.history['best_fitness']),
            'average_fitness': sum(self.history['best_fitness']) / len(self.history['best_fitness']),
            'total_iterations': len(self.history['iterations']),
            'total_time': sum(self.history['execution_time'])
        }