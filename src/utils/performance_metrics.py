"""
性能指标计算工具
"""

import numpy as np
from typing import List, Dict, Tuple, Any


class PerformanceMetrics:
    """性能指标计算类"""

    @staticmethod
    def calculate_igd(pareto_front: List, reference_front: List) -> float:
        """计算反向世代距离 (IGD)"""
        if not pareto_front or not reference_front:
            return float('inf')

        total_distance = 0
        for ref_point in reference_front:
            min_distance = min(
                np.linalg.norm(np.array(ref_point) - np.array(sol))
                for sol in pareto_front
            )
            total_distance += min_distance

        return total_distance / len(reference_front)

    @staticmethod
    def calculate_hypervolume(pareto_front: List, reference_point: List) -> float:
        """计算超体积指标"""
        if not pareto_front:
            return 0.0

        # 简化实现 - 实际应用中可能需要更复杂的计算
        front = np.array(pareto_front)
        ref = np.array(reference_point)

        # 归一化
        normalized_front = front / ref
        volume = 1.0

        for point in normalized_front:
            volume *= np.prod(1 - point)

        return 1 - volume

    @staticmethod
    def calculate_spacing(pareto_front: List) -> float:
        """计算分布均匀性指标"""
        if len(pareto_front) <= 1:
            return 0.0

        distances = []
        for i, sol1 in enumerate(pareto_front):
            min_dist = float('inf')
            for j, sol2 in enumerate(pareto_front):
                if i != j:
                    dist = np.linalg.norm(np.array(sol1) - np.array(sol2))
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)

        avg_distance = np.mean(distances)
        spacing = np.sqrt(sum((d - avg_distance) ** 2 for d in distances) / len(distances))
        return spacing

    @staticmethod
    def calculate_multi_objective_metrics(solutions: List, reference_point: List) -> Dict[str, float]:
        """计算多目标性能指标"""
        if not solutions:
            return {
                'igd': float('inf'),
                'hypervolume': 0.0,
                'spacing': 0.0,
                'number_of_solutions': 0
            }

        # 简化参考前沿 - 实际应用中应该使用真实的参考前沿
        ref_front = [[0.1, 0.1, 0.1]]  # 理想点附近

        return {
            'igd': PerformanceMetrics.calculate_igd(solutions, ref_front),
            'hypervolume': PerformanceMetrics.calculate_hypervolume(solutions, reference_point),
            'spacing': PerformanceMetrics.calculate_spacing(solutions),
            'number_of_solutions': len(solutions)
        }