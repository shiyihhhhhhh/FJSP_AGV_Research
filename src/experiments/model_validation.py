import time
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

from src.models.fjsp_agv_model import FJSPAGVModel
from src.models.model_validator import ModelValidator
from src.utils.data_generator import DataGenerator
from src.config.model_config import ModelConfig


class ModelValidationExperiment:
    """模型验证实验"""

    def __init__(self):
        self.results = {}

    def run_single_instance(self, config: ModelConfig) -> Dict[str, Any]:
        """运行单个实例验证"""
        print(f"🔬 运行实例验证: {config.n_jobs}工件, {config.n_machines}机器, {config.n_agvs}AGV")

        # 生成数据
        test_data = DataGenerator.generate_from_config(config)

        # 创建并求解模型
        model = FJSPAGVModel(test_data)
        model.create_variables()
        model.set_objective()
        model.add_constraints()

        success = model.solve(time_limit=300)

        if success:
            # 验证解
            validator = ModelValidator(model)
            validation_results = validator.validate_all()

            # 记录结果
            instance_key = f"{config.n_jobs}_{config.n_machines}_{config.n_agvs}"
            self.results[instance_key] = {
                'solution': model.solution,
                'performance_metrics': model.performance_metrics,
                'validation_results': validation_results,
                'solve_time': model.solve_time
            }

            print(f"✅ 实例 {instance_key} 验证完成")
            return self.results[instance_key]
        else:
            print(f"❌ 实例 {instance_key} 求解失败")
            return {}

    def run_multiple_instances(self, configs: List[ModelConfig]) -> Dict[str, Any]:
        """运行多个实例验证"""
        overall_results = {}

        for config in configs:
            result = self.run_single_instance(config)
            overall_results[f"{config.n_jobs}_{config.n_machines}_{config.n_agvs}"] = result

        return overall_results

    def generate_report(self) -> str:
        """生成验证报告"""
        report = ["📊 模型验证实验报告", "=" * 50]

        for instance, result in self.results.items():
            report.append(f"\n实例: {instance}")
            if result:
                report.append(f"  完工时间: {result['solution']['makespan']:.2f}")
                report.append(f"  总能耗: {result['solution']['energy']:.2f}")
                report.append(f"  总成本: {result['solution']['cost']:.2f}")
                report.append(f"  求解时间: {result['solve_time']:.2f}秒")
                report.append(f"  验证结果: {result['validation_results']}")
            else:
                report.append("  求解失败")

        return "\n".join(report)

    def plot_performance_comparison(self, save_path: str = None):
        """绘制性能比较图"""
        if not self.results:
            print("无结果数据")
            return

        instances = list(self.results.keys())
        makespans = [result['solution']['makespan'] for result in self.results.values() if result]
        energies = [result['solution']['energy'] for result in self.results.values() if result]
        costs = [result['solution']['cost'] for result in self.results.values() if result]
        solve_times = [result['solve_time'] for result in self.results.values() if result]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 完工时间比较
        ax1.bar(instances, makespans, color='skyblue')
        ax1.set_title('完工时间比较')
        ax1.set_ylabel('完工时间')
        ax1.tick_params(axis='x', rotation=45)

        # 总能耗比较
        ax2.bar(instances, energies, color='lightgreen')
        ax2.set_title('总能耗比较')
        ax2.set_ylabel('总能耗')
        ax2.tick_params(axis='x', rotation=45)

        # 总成本比较
        ax3.bar(instances, costs, color='lightcoral')
        ax3.set_title('总成本比较')
        ax3.set_ylabel('总成本')
        ax3.tick_params(axis='x', rotation=45)

        # 求解时间比较
        ax4.bar(instances, solve_times, color='gold')
        ax4.set_title('求解时间比较')
        ax4.set_ylabel('求解时间 (秒)')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 性能比较图已保存至: {save_path}")

        plt.show()