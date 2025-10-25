#!/usr/bin/env python3
"""
简化版模型测试脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"📁 项目根目录: {project_root}")
print(f"📁 当前工作目录: {os.getcwd()}")

try:
    print("🔄 尝试导入模块...")
    from src.models.fjsp_agv_model import FJSPAGVModel
    from src.utils.data_generator import DataGenerator
    from src.config.model_config import ModelConfig

    print("✅ 模块导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("🔍 检查目录结构...")

    # 检查目录结构
    for root, dirs, files in os.walk(project_root):
        level = root.replace(str(project_root), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py'):
                print(f'{subindent}{file}')
    sys.exit(1)


def simple_test():
    """简单测试"""
    print("\n🔬 开始简单测试...")

    # 创建配置
    config = ModelConfig(n_jobs=2, n_machines=2, n_agvs=1)

    # 生成测试数据
    test_data = DataGenerator.generate_from_config(config)
    print(f"✅ 测试数据生成成功: {test_data['n']}工件, {test_data['m']}机器, {test_data['v']}AGV")

    # 创建模型
    model = FJSPAGVModel(test_data)
    model.create_variables()
    model.set_objective()
    model.add_constraints()
    print("✅ 模型创建成功")

    # 求解模型
    success = model.solve(time_limit=60)

    if success:
        print("✅ 模型求解成功！")
        print(f"   完工时间: {model.solution['makespan']:.2f}")
        print(f"   总能耗: {model.solution['energy']:.2f}")
        print(f"   总成本: {model.solution['cost']:.2f}")
        return True
    else:
        print("❌ 模型求解失败")
        return False


if __name__ == "__main__":
    # 确保目录存在
    os.makedirs('results', exist_ok=True)

    # 运行测试
    success = simple_test()

    if success:
        print("\n🎉 简单测试通过！")
        print("   模型基本功能正常，可以进行更全面的测试。")
    else:
        print("\n❌ 简单测试失败")
        print("   需要检查模型配置和求解器设置。")

    sys.exit(0 if success else 1)