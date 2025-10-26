"""
强化学习算法基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRLAlgorithm(ABC):
    """强化学习算法基类"""

    def __init__(self, state_space: List, action_space: List, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}

    @abstractmethod
    def choose_action(self, state: Any, exploration_rate: float = 0.1) -> Any:
        """选择动作"""
        pass

    @abstractmethod
    def update(self, state: Any, action: Any, reward: float, next_state: Any):
        """更新Q值"""
        pass

    @abstractmethod
    def get_best_action(self, state: Any) -> Any:
        """获取最佳动作"""
        pass