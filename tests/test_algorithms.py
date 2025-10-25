import unittest
import sys
import os

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.algorithms.metaheuristic.nsga2 import NSGA2
from src.algorithms.reinforcement_learning.q_learning import QLearning


class TestNSGA2(unittest.TestCase):
    """NSGA2算法测试"""

    def setUp(self):
        """测试前准备"""
        config = {
            'population_size': 20,
            'max_generations': 10,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2
        }
        self.algorithm = NSGA2(config)

    def test_initialization(self):
        """测试初始化"""
        self.algorithm.initialize_population()

        self.assertEqual(len(self.algorithm.population), 20)
        self.assertIsInstance(self.algorithm.population[0], dict)

    def test_evaluation(self):
        """测试评估函数"""
        self.algorithm.initialize_population()
        fitness = self.algorithm.evaluate_population(self.algorithm.population)

        self.assertEqual(len(fitness), 20)
        self.assertIn('makespan', fitness[0])
        self.assertIn('energy', fitness[0])
        self.assertIn('cost', fitness[0])


class TestQLearning(unittest.TestCase):
    """Q-Learning算法测试"""

    def setUp(self):
        """测试前准备"""
        config = {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon': 0.1,
            'actions': ['action1', 'action2', 'action3']
        }
        self.algorithm = QLearning(config)

    def test_action_selection(self):
        """测试动作选择"""
        state = 'test_state'
        action = self.algorithm.select_action(state)

        self.assertIn(action, ['action1', 'action2', 'action3'])

    def test_q_update(self):
        """测试Q值更新"""
        state = 'state1'
        action = 'action1'
        reward = 1.0
        next_state = 'state2'

        # 初始Q值应该为0
        state_key = self.algorithm.state_to_key(state)
        if state_key in self.algorithm.q_table:
            initial_q = self.algorithm.q_table[state_key][action]
        else:
            initial_q = 0.0

        self.algorithm.update(state, action, reward, next_state)

        # 更新后Q值应该变化
        new_q = self.algorithm.q_table[state_key][action]
        self.assertNotEqual(initial_q, new_q)


if __name__ == '__main__':
    unittest.main()