"""
Q-Learning 实现 - 针对FJSP-AGV问题的自适应模块
"""

import numpy as np
from typing import Any, Dict, List, Tuple
import random
from .base_rl_algorithm import BaseRLAlgorithm

class QLearning(BaseRLAlgorithm):
    """Q-Learning 算法 - 针对FJSP-AGV问题的自适应优化"""

    def __init__(self, state_space: List, action_space: List,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 exploration_rate: float = 0.1):
        super().__init__(state_space, action_space, learning_rate, discount_factor)
        self.exploration_rate = exploration_rate
        self.initialize_q_table()

        # 学习历史记录
        self.learning_history = {
            'states_visited': [],
            'actions_taken': [],
            'rewards_received': [],
            'q_table_updates': 0
        }

    def initialize_q_table(self):
        """初始化Q表"""
        for state in self.state_space:
            self.q_table[state] = {}
            for action in self.action_space:
                # 使用小的随机值初始化，避免初始偏好
                self.q_table[state][action] = random.uniform(0, 0.1)

    def choose_action(self, state: Any, exploration_rate: float = None) -> Any:
        """ε-贪婪策略选择动作"""
        if exploration_rate is None:
            exploration_rate = self.exploration_rate

        if random.random() < exploration_rate:
            # 探索：随机选择动作
            action = random.choice(self.action_space)
            self.learning_history['states_visited'].append(state)
            self.learning_history['actions_taken'].append(action)
            return action
        else:
            # 利用：选择Q值最大的动作
            return self.get_best_action(state)

    def get_best_action(self, state: Any) -> Any:
        """获取最佳动作"""
        if state not in self.q_table:
            # 如果状态未见过，随机选择动作
            return random.choice(self.action_space)

        q_values = self.q_table[state]
        max_q = max(q_values.values())

        # 如果有多个相同Q值的动作，随机选择一个
        best_actions = [action for action, q_value in q_values.items() if q_value == max_q]
        action = random.choice(best_actions)

        self.learning_history['states_visited'].append(state)
        self.learning_history['actions_taken'].append(action)
        return action

    def update(self, state: Any, action: Any, reward: float, next_state: Any):
        """更新Q值"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.action_space}

        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.action_space}

        current_q = self.q_table[state].get(action, 0.0)
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0

        # Q-Learning 更新公式
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

        self.learning_history['rewards_received'].append(reward)
        self.learning_history['q_table_updates'] += 1

    def get_q_value(self, state: Any, action: Any) -> float:
        """获取Q值"""
        if state not in self.q_table:
            return 0.0
        return self.q_table[state].get(action, 0.0)

    def get_policy(self) -> Dict[Any, Any]:
        """获取当前策略"""
        policy = {}
        for state in self.q_table:
            policy[state] = self.get_best_action(state)
        return policy

    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        if not self.learning_history['rewards_received']:
            return {
                'total_updates': 0,
                'average_reward': 0,
                'exploration_rate': self.exploration_rate,
                'states_visited': len(set(self.learning_history['states_visited']))
            }

        return {
            'total_updates': self.learning_history['q_table_updates'],
            'average_reward': np.mean(self.learning_history['rewards_received']),
            'max_reward': max(self.learning_history['rewards_received']),
            'min_reward': min(self.learning_history['rewards_received']),
            'exploration_rate': self.exploration_rate,
            'states_visited': len(set(self.learning_history['states_visited'])),
            'unique_actions': len(set(self.learning_history['actions_taken']))
        }

    def decrease_exploration_rate(self, decay_factor: float = 0.99):
        """降低探索率"""
        self.exploration_rate *= decay_factor
        self.exploration_rate = max(0.01, self.exploration_rate)  # 保持最小探索率