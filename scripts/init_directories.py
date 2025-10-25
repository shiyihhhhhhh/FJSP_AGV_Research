#!/usr/bin/env python3
"""
初始化项目目录结构
"""

import os
from pathlib import Path


def init_directories():
    """初始化项目目录"""
    directories = [
        'results',
        'data/raw',
        'data/processed',
        'data/results',
        'notebooks',
        'logs'
    ]

    base_dir = Path(__file__).parent.parent

    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {full_path}")

    # 创建 .gitkeep 文件以确保空目录被版本控制
    for dir_path in directories:
        gitkeep_file = base_dir / dir_path / '.gitkeep'
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            print(f"✅ 创建文件: {gitkeep_file}")


if __name__ == "__main__":
    print("📁 初始化项目目录结构...")
    init_directories()
    print("🎉 目录初始化完成！")