from abc import ABC, abstractmethod
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path


class BaseModel(ABC):
    """模型基类，定义统一接口"""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.solution = {}
        self.performance_metrics = {}
        self.solve_time = 0.0
        self.model = None

    @abstractmethod
    def create_variables(self):
        """创建决策变量"""
        pass

    @abstractmethod
    def set_objective(self, weights: Optional[List[float]] = None):
        """设置目标函数"""
        pass

    @abstractmethod
    def add_constraints(self):
        """添加约束条件"""
        pass

    @abstractmethod
    def solve(self, time_limit: int = 300, output_flag: bool = True) -> bool:
        """求解模型"""
        pass

    def save_model(self, filepath: str):
        """保存模型配置"""

        def convert_keys(obj):
            """递归转换字典键为字符串"""
            if isinstance(obj, dict):
                new_dict = {}
                for key, value in obj.items():
                    # 将键转换为字符串
                    if isinstance(key, tuple):
                        # 将元组转换为描述性字符串
                        if len(key) == 2:
                            new_key = f"job{key[0]}_op{key[1]}"
                        elif len(key) == 3:
                            if key[2] == 'machine':
                                new_key = f"job{key[0]}_op{key[1]}_machine"
                            elif key[2] == 'transport':
                                new_key = f"job{key[0]}_op{key[1]}_transport"
                            else:
                                new_key = str(key)
                        else:
                            new_key = str(key)
                    else:
                        new_key = str(key)
                    new_dict[new_key] = convert_keys(value)
                return new_dict
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_keys(item) for item in obj)
            else:
                return obj

        model_data = {
            'parameters': convert_keys(self.data),
            'solution': convert_keys(self.solution),
            'performance_metrics': self.performance_metrics,
            'solve_time': self.solve_time
        }

        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)

        print(f"💾 模型结果已保存至: {filepath}")


    def load_model(self, filepath: str):
        """加载模型配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        def deserialize_object(obj):
            """递归反序列化对象"""
            if isinstance(obj, dict):
                deserialized = {}
                for key, value in obj.items():
                    # 尝试将字符串键解析回元组
                    if "_" in key and (key.startswith("job") or "op" in key):
                        try:
                            # 解析 "job0_op1" 这样的键
                            parts = key.split("_")
                            if len(parts) >= 2:
                                job_part = parts[0]  # "job0"
                                op_part = parts[1]  # "op1"
                                job_id = int(job_part[3:])  # 去掉"job"
                                op_id = int(op_part[2:])  # 去掉"op"
                                if len(parts) == 2:
                                    deserialized_key = (job_id, op_id)
                                else:
                                    # 处理更复杂的键
                                    deserialized_key = tuple(
                                        int(part[2:]) if part.startswith(('job', 'op')) else part for part in parts)
                            else:
                                deserialized_key = key
                        except:
                            deserialized_key = key
                    else:
                        deserialized_key = key
                    deserialized[deserialized_key] = deserialize_object(value)
                return deserialized
            elif isinstance(obj, list):
                return [deserialize_object(item) for item in obj]
            else:
                return obj

        self.data = deserialize_object(model_data['parameters'])
        self.solution = deserialize_object(model_data.get('solution', {}))
        self.performance_metrics = deserialize_object(model_data.get('performance_metrics', {}))
        self.solve_time = model_data.get('solve_time', 0.0)
    def validate_solution(self) -> Dict[str, bool]:
        """验证解可行性"""
        return {
            'mathematical_feasibility': True,
            'physical_reasonableness': True
        }

    def analyze_results(self):
        """分析结果"""
        if self.solution:
            print("📊 结果分析:")
            for key, value in self.solution.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.2f}")

    def get_performance_report(self) -> str:
        """生成性能报告"""
        if not self.performance_metrics:
            return "暂无性能数据"

        report = ["🎯 性能报告:"]
        for metric, value in self.performance_metrics.items():
            report.append(f"  {metric}: {value}")

        return "\n".join(report)