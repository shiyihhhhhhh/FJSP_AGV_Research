from abc import ABC, abstractmethod
from typing import Dict, Any, List
import time


class BaseAlgorithm(ABC):
    """算法基类"""

    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.solution = {}
        self.convergence_data = []
        self.runtime = 0.0

    @abstractmethod
    def initialize(self):
        """初始化算法"""
        pass

    @abstractmethod
    def run(self):
        """运行算法"""
        pass

    @abstractmethod
    def get_solution(self) -> Dict:
        """获取解"""
        pass

    def evaluate_solution(self, solution: Dict) -> float:
        """评估解的质量"""
        # 这里可以使用模型的评估方法
        if hasattr(self.model, 'evaluate_solution'):
            return self.model.evaluate_solution(solution)
        else:
            # 默认评估方法
            return sum(solution.values())

    def save_results(self, filepath: str):
        """保存结果"""
        results = {
            'solution': self.solution,
            'convergence_data': self.convergence_data,
            'runtime': self.runtime,
            'config': self.config
        }

        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def plot_convergence(self, save_path: str = None):
        """绘制收敛曲线"""
        if not self.convergence_data:
            print("无收敛数据")
            return

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_data)
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title('算法收敛曲线')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()