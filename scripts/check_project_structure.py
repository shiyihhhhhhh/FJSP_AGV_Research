#!/usr/bin/env python3
"""
检查项目结构脚本
"""

import os
from pathlib import Path


def check_project_structure():
    """检查项目结构"""
    project_root = Path(__file__).parent.parent

    print("📁 项目结构检查")
    print("=" * 50)

    required_dirs = [
        'src',
        'src/models',
        'src/algorithms',
        'src/algorithms/metaheuristic',
        'src/algorithms/reinforcement_learning',
        'src/utils',
        'src/config',
        'src/experiments',
        'tests',
        'scripts',
        'results'
    ]

    required_files = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/models/fjsp_agv_model.py',
        'src/models/base_model.py',
        'src/models/model_validator.py',
        'src/utils/__init__.py',
        'src/utils/data_generator.py',
        'src/config/__init__.py',
        'src/config/model_config.py',
        'main.py'
    ]

    print("🔍 检查目录...")
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} - 目录不存在")

    print("\n🔍 检查文件...")
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - 文件不存在")

    # 检查Python路径
    print(f"\n🔍 Python路径:")
    import sys
    for i, path in enumerate(sys.path):
        print(f"   {i + 1}. {path}")


if __name__ == "__main__":
    check_project_structure()