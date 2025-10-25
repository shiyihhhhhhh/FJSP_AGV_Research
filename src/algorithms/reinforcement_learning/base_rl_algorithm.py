from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class BaseRLAlgorithm(ABC):
    """强化学习算法基类"""

    def __init__(self, config: Dict):
        self.config = config
        self.q_table = {}
        self.learning_rate = config.get('learning_rate', 0.1)
        self.discount_factor = config.get('discount_factor', 0.9)
        self.epsilon = config.get('epsilon', 0.1)
        self.episode_rewards = []

    @abstractmethod
    def select_action(self, state: Any) -> Any:
        """选择动作"""
        pass

    @abstractmethod
    def update(self, state: Any, action: Any, reward: float, next_state: Any):
        """更新Q值"""
        pass

    @abstractmethod
    def get_policy(self) -> Dict:
        """获取策略"""
        pass

    def state_to_key(self, state: Any) -> str:
        """将状态转换为字典键"""
        if isinstance(state, (list, tuple)):
            return str(tuple(state))
        elif isinstance(state, dict):
            return str(tuple(sorted(state.items())))
        else:
            return str(state)

    def save_q_table(self, filepath: str):
        """保存Q表"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.q_table, f, indent=2, ensure_ascii=False)

    def load_q_table(self, filepath: str):
        """加载Q表"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            self.q_table = json.load(f)