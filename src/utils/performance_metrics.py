from typing import Dict, Any, List
import numpy as np


class PerformanceMetrics:
    """性能指标计算"""

    @staticmethod
    def calculate_gaps(actual: Dict[str, float], theoretical: Dict[str, float]) -> Dict[str, float]:
        """计算与理论下界的差距"""
        gaps = {}
        for key in actual:
            if key in theoretical:
                gaps[key] = (actual[key] - theoretical[key]) / max(theoretical[key], 1e-6)
        return gaps

    @staticmethod
    def calculate_hypervolume(pareto_front: List[Dict[str, float]], reference: Dict[str, float]) -> float:
        """计算超体积指标"""
        if not pareto_front:
            return 0.0

        # 简化实现
        volume = 1.0
        for key in reference:
            min_val = min(solution[key] for solution in pareto_front)
            volume *= (reference[key] - min_val) / reference[key]

        return volume

    @staticmethod
    def calculate_spacing(pareto_front: List[Dict[str, float]]) -> float:
        """计算解集的间距"""
        if len(pareto_front) < 2:
            return 0.0

        distances = []
        for i, sol1 in enumerate(pareto_front):
            min_dist = float('inf')
            for j, sol2 in enumerate(pareto_front):
                if i != j:
                    dist = np.sqrt(sum((sol1[key] - sol2[key]) ** 2 for key in sol1))
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)

        avg_distance = np.mean(distances)
        spacing = np.sqrt(sum((d - avg_distance) ** 2 for d in distances) / len(distances))

        return spacing