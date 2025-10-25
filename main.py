#!/usr/bin/env python3
"""
FJSP-AGV集成调度研究主程序
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.models.fjsp_agv_model import FJSPAGVModel
    from src.models.model_validator import ModelValidator
    from src.utils.data_generator import DataGenerator
    from src.config.model_config import ModelConfig
    from src.config.algorithm_config import AlgorithmConfig
    from src.experiments.model_validation import ModelValidationExperiment
    from src.experiments.algorithm_comparison import AlgorithmComparisonExperiment
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("📁 请确保项目结构正确，src目录存在")
    sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FJSP-AGV集成调度研究')
    parser.add_argument('--mode', type=str, choices=['model', 'algorithm', 'experiment'],
                        default='model', help='运行模式: model(模型验证), algorithm(算法测试), experiment(完整实验)')
    parser.add_argument('--config', type=str, help='配置文件路径')

    args = parser.parse_args()

    if args.mode == 'model':
        run_model_validation()
    elif args.mode == 'algorithm':
        run_algorithm_test()
    elif args.mode == 'experiment':
        run_full_experiment()
    else:
        print("未知模式，使用 --help 查看帮助")


def run_model_validation():
    """运行模型验证"""
    print("🔬 运行模型验证...")

    # 确保目录存在
    os.makedirs('results', exist_ok=True)

    # 创建配置
    config = ModelConfig(n_jobs=3, n_machines=2, n_agvs=1)

    # 生成测试数据
    test_data = DataGenerator.generate_from_config(config)

    # 创建并求解模型
    model = FJSPAGVModel(test_data)
    model.create_variables()
    model.set_objective()
    model.add_constraints()

    if model.solve():
        print("✅ 模型求解成功！")

        # 验证解
        validator = ModelValidator(model)
        validation_results = validator.validate_all()
        print(validator.generate_validation_report())

        # 绘制调度图
        model.plot_schedule('results/schedule_gantt.png')

        # 绘制电量使用图
        validator.plot_battery_usage('results/battery_usage.png')

        # 保存模型结果
        model.save_model('results/model_solution.json')

    else:
        print("❌ 模型求解失败")


def run_algorithm_test():
    """运行算法测试"""
    print("🔧 运行算法测试...")

    # 创建模型配置
    model_config = ModelConfig(n_jobs=3, n_machines=2, n_agvs=1)

    # 创建算法配置
    algorithm_configs = {
        'NSGA2': {
            'population_size': 50,
            'max_generations': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2
        },
        'Hybrid': {
            'max_generations': 100,
            'metaheuristic': {
                'population_size': 50,
                'crossover_rate': 0.8,
                'mutation_rate': 0.2
            },
            'reinforcement_learning': {
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'epsilon': 0.1,
                'actions': ['intensify', 'diversify', 'balance']
            }
        }
    }

    # 运行算法比较
    experiment = AlgorithmComparisonExperiment()
    results = experiment.run_comparison(model_config, algorithm_configs)

    # 生成报告
    print(experiment.generate_comparison_report())

    # 绘制收敛曲线
    experiment.plot_convergence_comparison('results/convergence_comparison.png')


def run_full_experiment():
    """运行完整实验"""
    print("🎯 运行完整实验...")

    # 1. 模型验证实验
    print("\n" + "=" * 50)
    print("阶段1: 模型验证实验")
    print("=" * 50)

    model_experiment = ModelValidationExperiment()

    # 定义多个测试实例
    test_configs = [
        ModelConfig(n_jobs=2, n_machines=2, n_agvs=1, operations_per_job=[2, 2]),
        ModelConfig(n_jobs=3, n_machines=2, n_agvs=1, operations_per_job=[2, 2, 2]),
        ModelConfig(n_jobs=4, n_machines=3, n_agvs=1, operations_per_job=[2, 2, 2, 2]),
    ]

    model_results = model_experiment.run_multiple_instances(test_configs)
    print(model_experiment.generate_report())
    model_experiment.plot_performance_comparison('results/model_performance_comparison.png')

    # 2. 算法比较实验
    print("\n" + "=" * 50)
    print("阶段2: 算法比较实验")
    print("=" * 50)

    algorithm_experiment = AlgorithmComparisonExperiment()

    # 使用中等规模的实例
    algo_model_config = ModelConfig(n_jobs=5, n_machines=3, n_agvs=2, operations_per_job=[3, 3, 3, 3, 3])

    algorithm_configs = {
        'NSGA2': {
            'population_size': 100,
            'max_generations': 200,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2
        },
        'Hybrid': {
            'max_generations': 200,
            'metaheuristic': {
                'population_size': 100,
                'crossover_rate': 0.8,
                'mutation_rate': 0.2
            },
            'reinforcement_learning': {
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'epsilon': 0.1,
                'actions': ['intensify', 'diversify', 'balance']
            }
        }
    }

    algo_results = algorithm_experiment.run_comparison(algo_model_config, algorithm_configs)
    print(algorithm_experiment.generate_comparison_report())
    algorithm_experiment.plot_convergence_comparison('results/algorithm_convergence_comparison.png')

    print("\n🎉 完整实验完成！")
    print("📁 结果已保存至 results/ 目录")


if __name__ == "__main__":
    main()

