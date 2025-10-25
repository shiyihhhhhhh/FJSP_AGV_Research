#!/usr/bin/env python3
"""
模型完整性测试脚本
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models.fjsp_agv_model import FJSPAGVModel
    from src.models.model_validator import ModelValidator
    from src.utils.data_generator import DataGenerator
    from src.config.model_config import ModelConfig
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("📁 当前工作目录:", os.getcwd())
    print("📁 项目根目录:", project_root)
    print("📁 Python路径:")
    for path in sys.path:
        print(f"   {path}")
    sys.exit(1)


def comprehensive_model_test():
    """综合模型测试"""
    print("🔍 开始综合模型测试")
    print("=" * 60)

    test_results = {}

    # 测试用例
    test_cases = [
        ("小型实例", ModelConfig(n_jobs=2, n_machines=2, n_agvs=1, operations_per_job=[2, 2])),
        ("标准实例", ModelConfig(n_jobs=3, n_machines=2, n_agvs=1, operations_per_job=[2, 2, 2])),
        ("中等实例", ModelConfig(n_jobs=4, n_machines=3, n_agvs=1, operations_per_job=[2, 2, 2, 2])),
    ]

    for case_name, config in test_cases:
        print(f"\n📊 测试案例: {case_name}")
        print(f"   配置: {config.n_jobs}工件, {config.n_machines}机器, {config.n_agvs}AGV")

        try:
            # 生成测试数据
            test_data = DataGenerator.generate_from_config(config)

            # 创建模型
            model = FJSPAGVModel(test_data)
            model.create_variables()
            model.set_objective()
            model.add_constraints()

            # 求解模型
            start_time = time.time()
            success = model.solve(time_limit=120)
            solve_time = time.time() - start_time

            if success:
                # 验证解
                validator = ModelValidator(model)
                validation_results = validator.validate_all()

                # 记录结果
                test_results[case_name] = {
                    'success': True,
                    'solve_time': solve_time,
                    'solution_quality': {
                        'makespan': model.solution['makespan'],
                        'energy': model.solution['energy'],
                        'cost': model.solution['cost']
                    },
                    'validation': validation_results,
                    'performance_gaps': model.performance_metrics
                }

                print(f"   ✅ 求解成功 - 用时: {solve_time:.2f}s")
                print(f"      完工时间: {model.solution['makespan']:.2f}")
                print(f"      总能耗: {model.solution['energy']:.2f}")
                print(f"      总成本: {model.solution['cost']:.2f}")

            else:
                test_results[case_name] = {
                    'success': False,
                    'solve_time': solve_time,
                    'error': '求解失败'
                }
                print(f"   ❌ 求解失败")

        except Exception as e:
            test_results[case_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ 出错: {e}")

    return test_results


def generate_test_report(test_results):
    """生成测试报告"""
    print("\n" + "=" * 60)
    print("📊 模型完整性测试报告")
    print("=" * 60)

    total_cases = len(test_results)
    successful_cases = sum(1 for result in test_results.values() if result['success'])

    print(f"测试案例总数: {total_cases}")
    print(f"成功案例数: {successful_cases}")
    print(f"成功率: {successful_cases / total_cases * 100:.1f}%")

    print("\n详细结果:")
    for case_name, result in test_results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"\n{case_name}: {status}")

        if result['success']:
            print(f"   求解时间: {result['solve_time']:.2f}s")
            print(f"   完工时间: {result['solution_quality']['makespan']:.2f}")
            print(f"   总能耗: {result['solution_quality']['energy']:.2f}")
            print(f"   总成本: {result['solution_quality']['cost']:.2f}")

            # 验证结果
            validation = result['validation']
            print(f"   验证结果:")
            for check, passed in validation.items():
                status = "通过" if passed else "失败"
                print(f"     - {check}: {status}")
        else:
            print(f"   错误: {result.get('error', '未知错误')}")

    # 总体评估
    if successful_cases == total_cases:
        print("\n🎉 模型完整性测试完全通过！")
        print("   模型在各个方面都表现良好，可以用于进一步研究。")
    elif successful_cases >= total_cases * 0.7:
        print("\n⚠️ 模型完整性测试基本通过。")
        print("   模型可用于研究，但建议关注失败案例。")
    else:
        print("\n❌ 模型完整性测试未通过。")
        print("   需要修复模型中的问题。")

    return successful_cases == total_cases


if __name__ == "__main__":
    # 确保目录存在
    os.makedirs('results', exist_ok=True)

    # 运行测试
    results = comprehensive_model_test()

    # 生成报告
    all_passed = generate_test_report(results)

    # 保存详细结果
    import json

    with open('results/model_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 详细结果已保存至: results/model_test_results.json")

    # 退出码
    sys.exit(0 if all_passed else 1)