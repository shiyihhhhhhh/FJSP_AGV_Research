import unittest
import sys
import os

# 添加src到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.fjsp_agv_model import FJSPAGVModel
from src.utils.data_generator import DataGenerator
from src.config.model_config import ModelConfig


class TestFJSPAGVModel(unittest.TestCase):
    """FJSP-AGV模型测试"""

    def setUp(self):
        """测试前准备"""
        config = ModelConfig(n_jobs=2, n_machines=2, n_agvs=1)
        test_data = DataGenerator.generate_from_config(config)
        self.model = FJSPAGVModel(test_data)

    def test_model_creation(self):
        """测试模型创建"""
        self.model.create_variables()
        self.model.set_objective()
        self.model.add_constraints()

        # 检查变量是否创建
        self.assertIn('C_max', self.model.variables)
        self.assertIn('s', self.model.variables)
        self.assertIn('x', self.model.variables)

    def test_theoretical_bounds(self):
        """测试理论下界计算"""
        bounds = self.model.calculate_theoretical_bounds()

        self.assertIn('makespan', bounds)
        self.assertIn('energy', bounds)
        self.assertIn('cost', bounds)

        # 检查下界是否合理
        self.assertGreater(bounds['makespan'], 0)
        self.assertGreater(bounds['energy'], 0)
        self.assertGreater(bounds['cost'], 0)

    def test_small_instance_solve(self):
        """测试小规模实例求解"""
        self.model.create_variables()
        self.model.set_objective()
        self.model.add_constraints()

        success = self.model.solve(time_limit=30)

        # 小规模实例应该能求解
        self.assertTrue(success)

        if success:
            # 检查解的存在性
            self.assertIsNotNone(self.model.solution)
            self.assertIn('makespan', self.model.solution)
            self.assertIn('energy', self.model.solution)
            self.assertIn('cost', self.model.solution)


if __name__ == '__main__':
    unittest.main()