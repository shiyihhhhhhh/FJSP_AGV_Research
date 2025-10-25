import time
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.hybrid_algorithm import HybridAlgorithm
from src.algorithms.metaheuristic.nsga2 import NSGA2
from src.utils.data_generator import DataGenerator
from src.config.model_config import ModelConfig
from src.config.algorithm_config import AlgorithmConfig


class AlgorithmComparisonExperiment:
    """算法比较实验"""

    def __init__(self):
        self.results = {}
        self.algorithms = {
            'NSGA2': NSGA2,
            'Hybrid': HybridAlgorithm
        }

    def run_algorithm(self, algorithm_name: str, model, config: Dict) -> Dict[str, Any]:
        """运行单个算法"""
        print(f"🔧 运行算法: {algorithm_name}")

        start_time = time.time()

        if algorithm_name == 'Hybrid':
            algorithm = self.algorithms[algorithm_name](model, config)
        else:
            algorithm = self.algorithms[algorithm_name](config)

        algorithm.initialize()
        algorithm.run()
        solution = algorithm.get_solution()

        runtime = time.time() - start_time

        return {
            'solution': solution,
            'convergence': algorithm.convergence_data,
            'runtime': runtime
        }

    def run_comparison(self, model_config: ModelConfig, algorithm_configs: Dict[str, Dict]) -> Dict[str, Any]:
        """运行算法比较"""
        # 生成测试数据
        test_data = DataGenerator.generate_from_config(model_config)

        # 为每个算法运行
        comparison_results = {}

        for algo_name, algo_config in algorithm_configs.items():
            print(f"\n🎯 运行 {algo_name} 算法...")

            # 创建模型实例（混合算法需要）
            from src.models.fjsp_agv_model import FJSPAGVModel
            model = FJSPAGVModel(test_data)

            result = self.run_algorithm(algo_name, model, algo_config)
            comparison_results[algo_name] = result

            print(f"✅ {algo_name} 完成, 运行时间: {result['runtime']:.2f}秒")

        self.results = comparison_results
        return comparison_results

    def plot_convergence_comparison(self, save_path: str = None):
        """绘制收敛曲线比较"""
        if not self.results:
            print("无结果数据")
            return

        plt.figure(figsize=(10, 6))

        for algo_name, result in self.results.items():
            if result['convergence']:
                plt.plot(result['convergence'], label=algo_name)

        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title('算法收敛曲线比较')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 收敛曲线比较图已保存至: {save_path}")

        plt.show()

    def generate_comparison_report(self) -> str:
        """生成比较报告"""
        report = ["📊 算法比较实验报告", "=" * 50]

        for algo_name, result in self.results.items():
            report.append(f"\n算法: {algo_name}")
            report.append(f"  运行时间: {result['runtime']:.2f}秒")
            report.append(f"  最终解: {result['solution']}")
            report.append(f"  收敛代数: {len(result['convergence'])}")

        return "\n".join(report)