import itertools
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

from src.models.fjsp_agv_model import FJSPAGVModel
from src.utils.data_generator import DataGenerator
from src.config.model_config import ModelConfig


class ParameterTuningExperiment:
    """参数调优实验"""

    def __init__(self):
        self.results = {}

    def tune_objective_weights(self, model_config: ModelConfig, weight_combinations: List[List[float]]) -> Dict[
        str, Any]:
        """调优目标函数权重"""
        print("🎯 开始目标权重调优实验")

        test_data = DataGenerator.generate_from_config(model_config)
        results = {}

        for i, weights in enumerate(weight_combinations):
            print(f"测试权重组合 {i + 1}/{len(weight_combinations)}: {weights}")

            model = FJSPAGVModel(test_data)
            model.create_variables()
            model.set_objective(weights)
            model.add_constraints()

            if model.solve(time_limit=60):
                results[str(weights)] = {
                    'solution': model.solution,
                    'performance_metrics': model.performance_metrics,
                    'solve_time': model.solve_time
                }

        self.results['weight_tuning'] = results
        return results

    def tune_algorithm_parameters(self, algorithm_class, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """调优算法参数"""
        print("🔧 开始算法参数调优实验")

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        results = {}

        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            print(f"测试参数组合 {i + 1}/{len(param_combinations)}: {params}")

            # 创建算法实例并运行
            algorithm = algorithm_class(params)
            result = algorithm.run()

            results[str(params)] = {
                'result': result,
                'best_solution': algorithm.best_solution,
                'convergence': algorithm.convergence
            }

        self.results['algorithm_tuning'] = results
        return results

    def plot_weight_sensitivity(self, save_path: str = None):
        """绘制权重敏感性分析"""
        if 'weight_tuning' not in self.results:
            print("无权重调优数据")
            return

        weight_results = self.results['weight_tuning']

        weights = []
        makespans = []
        energies = []
        costs = []

        for weight_str, result in weight_results.items():
            weight = eval(weight_str)  # 将字符串转换回列表
            weights.append(weight)
            makespans.append(result['solution']['makespan'])
            energies.append(result['solution']['energy'])
            costs.append(result['solution']['cost'])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # 完工时间敏感性
        ax1.scatter([w[0] for w in weights], makespans, alpha=0.7)
        ax1.set_xlabel('完工时间权重')
        ax1.set_ylabel('完工时间')
        ax1.set_title('完工时间权重敏感性')

        # 能耗敏感性
        ax2.scatter([w[1] for w in weights], energies, alpha=0.7)
        ax2.set_xlabel('能耗权重')
        ax2.set_ylabel('总能耗')
        ax2.set_title('能耗权重敏感性')

        # 成本敏感性
        ax3.scatter([w[2] for w in weights], costs, alpha=0.7)
        ax3.set_xlabel('成本权重')
        ax3.set_ylabel('总成本')
        ax3.set_title('成本权重敏感性')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def find_best_parameters(self) -> Dict[str, Any]:
        """找到最佳参数"""
        best_params = {}

        if 'weight_tuning' in self.results:
            weight_results = self.results['weight_tuning']
            best_weight = min(weight_results.items(),
                              key=lambda x: sum(x[1]['solution'].values()))
            best_params['weights'] = eval(best_weight[0])

        if 'algorithm_tuning' in self.results:
            algo_results = self.results['algorithm_tuning']
            best_algo_params = min(algo_results.items(),
                                   key=lambda x: sum(x[1]['best_solution'].values()))
            best_params['algorithm'] = eval(best_algo_params[0])

        return best_params