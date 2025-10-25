from typing import Dict, Any, List
import numpy as np
from .base_rl_algorithm import BaseRLAlgorithm


class QLearning(BaseRLAlgorithm):
    """Q-Learning 算法"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.actions = config.get('actions', ['action1', 'action2', 'action3'])

    def select_action(self, state: Any) -> Any:
        """ε-greedy策略选择动作"""
        state_key = self.state_to_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}

        if np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.choice(self.actions)
        else:
            # 利用：选择Q值最大的动作
            q_values = self.q_table[state_key]
            max_q = max(q_values.values())
            # 从具有最大Q值的动作中随机选择一个
            best_actions = [action for action, q in q_values.items() if q == max_q]
            return np.random.choice(best_actions)

    def update(self, state: Any, action: Any, reward: float, next_state: Any):
        """更新Q值"""
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        # 初始化Q表
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {action: 0.0 for action in self.actions}

        # Q-learning更新公式
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values())

        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

    def get_policy(self) -> Dict:
        """获取最优策略"""
        policy = {}
        for state, actions in self.q_table.items():
            best_action = max(actions, key=actions.get)
            policy[state] = best_action
        return policy

    def train(self, episodes: int, environment):
        """训练Q-learning代理"""
        for episode in range(episodes):
            state = environment.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done = environment.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            self.episode_rewards.append(total_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")