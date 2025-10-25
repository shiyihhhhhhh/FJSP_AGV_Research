import unittest
import sys
import os

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.data_generator import DataGenerator
from src.config.model_config import ModelConfig


class TestDataGenerator(unittest.TestCase):
    """数据生成器测试"""

    def test_data_generation(self):
        """测试数据生成"""
        config = ModelConfig(n_jobs=3, n_machines=2, n_agvs=1)
        data = DataGenerator.generate_from_config(config)

        # 检查基本参数
        self.assertEqual(data['n'], 3)
        self.assertEqual(data['m'], 2)
        self.assertEqual(data['v'], 1)

        # 检查加工参数
        self.assertIn('p_jhi', data)
        self.assertIn('e_machine_jhi', data)
        self.assertIn('c_jhi', data)

        # 检查AGV参数
        self.assertIn('e_AGV_jh', data)
        self.assertIn('tt_jh', data)
        self.assertIn('q_jh', data)

    def test_specific_instance(self):
        """测试特定实例生成"""
        from typing import List
        data = DataGenerator.generate_specific_instance(
            n_jobs=2, n_machines=2, n_agvs=1, operations_per_job=[2, 2]
        )

        self.assertEqual(data['n'], 2)
        self.assertEqual(data['m'], 2)
        self.assertEqual(data['v'], 1)
        self.assertEqual(data['o_j'], [2, 2])


if __name__ == '__main__':
    unittest.main()