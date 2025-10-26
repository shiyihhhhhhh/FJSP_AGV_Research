"""
解决方案旋转器 - 负责子种群间的个体迁移和交换
"""

import random
from typing import List, Dict, Any
import numpy as np


class SolutionRotator:
    """解决方案旋转器"""

    def __init__(self):
        self.migration_strategies = {
            'elite': self._elite_migration,
            'random': self._random_migration,
            'ring': self._ring_migration,
            'fully_connected': self._fully_connected_migration,
            'adaptive': self._adaptive_migration
        }

        self.rotation_history = []

    def rotate_solutions(self, subpopulations: List[List], strategy: str = "elite",
                         migration_rate: float = 0.1) -> List[List]:
        """执行解决方案旋转"""
        if len(subpopulations) <= 1:
            return subpopulations

        strategy_func = self.migration_strategies.get(strategy, self._elite_migration)
        rotated_populations = strategy_func(subpopulations, migration_rate)

        # 记录旋转历史
        self.rotation_history.append({
            'strategy': strategy,
            'migration_rate': migration_rate,
            'subpopulations': len(subpopulations),
            'timestamp': self._get_timestamp()
        })

        return rotated_populations

    def _elite_migration(self, populations: List[List], migration_rate: float) -> List[List]:
        """精英迁移策略"""
        n_populations = len(populations)
        migration_count = max(1, int(len(populations[0]) * migration_rate))

        # 创建新种群列表（深拷贝）
        new_populations = [pop.copy() for pop in populations]

        for i in range(n_populations):
            source_idx = i
            target_idx = (i + 1) % n_populations

            if (len(new_populations[source_idx]) > migration_count and
                    len(new_populations[target_idx]) >= migration_count):
                # 选择源种群的精英
                elites = self._select_elites(new_populations[source_idx], migration_count)

                # 替换目标种群的最差个体
                self._replace_worst(new_populations[target_idx], elites)

        return new_populations

    def _random_migration(self, populations: List[List], migration_rate: float) -> List[List]:
        """随机迁移策略"""
        n_populations = len(populations)
        migration_count = max(1, int(len(populations[0]) * migration_rate))

        new_populations = [pop.copy() for pop in populations]

        for i in range(n_populations):
            source_idx = i
            target_idx = (i + 1) % n_populations

            if (len(new_populations[source_idx]) > migration_count and
                    len(new_populations[target_idx]) >= migration_count):

                # 随机选择迁移个体
                migrants = random.sample(new_populations[source_idx], migration_count)

                # 随机替换目标种群的个体
                indices_to_replace = random.sample(range(len(new_populations[target_idx])), migration_count)
                for idx, migrant in zip(indices_to_replace, migrants):
                    new_populations[target_idx][idx] = migrant

        return new_populations

    def _ring_migration(self, populations: List[List], migration_rate: float) -> List[List]:
        """环状迁移策略"""
        n_populations = len(populations)
        migration_count = max(1, int(len(populations[0]) * migration_rate))

        new_populations = [pop.copy() for pop in populations]

        # 创建环状拓扑迁移
        for i in range(n_populations):
            source = new_populations[i]
            target = new_populations[(i + 1) % n_populations]

            if len(source) > migration_count and len(target) >= migration_count:
                elites = self._select_elites(source, migration_count)
                self._replace_worst(target, elites)

        return new_populations

    def _fully_connected_migration(self, populations: List[List], migration_rate: float) -> List[List]:
        """全连接迁移策略"""
        n_populations = len(populations)
        migration_count = max(1, int(len(populations[0]) * migration_rate * 0.5))

        new_populations = [pop.copy() for pop in populations]

        # 每个种群向所有其他种群迁移
        for i in range(n_populations):
            for j in range(n_populations):
                if i != j:
                    if (len(new_populations[i]) > migration_count and
                            len(new_populations[j]) >= migration_count):
                        elites = self._select_elites(new_populations[i], migration_count)
                        self._replace_worst(new_populations[j], elites)

        return new_populations

    def _adaptive_migration(self, populations: List[List], migration_rate: float) -> List[List]:
        """自适应迁移策略"""
        # 基于种群多样性动态调整迁移策略
        diversities = [self._calculate_diversity(pop) for pop in populations]
        avg_diversity = np.mean(diversities)

        if avg_diversity < 0.1:
            # 低多样性，使用全连接迁移增加多样性
            return self._fully_connected_migration(populations, migration_rate)
        elif avg_diversity > 0.3:
            # 高多样性，使用精英迁移保持优良特性
            return self._elite_migration(populations, migration_rate)
        else:
            # 中等多样性，使用环状迁移平衡探索和利用
            return self._ring_migration(populations, migration_rate)

    def _select_elites(self, population: List, count: int) -> List:
        """选择精英个体"""
        if not population:
            return []

        # 按适应度排序（假设染色体有fitness属性）
        try:
            sorted_pop = sorted(population, key=lambda x: getattr(x, 'fitness', 0))
            return sorted_pop[:count]
        except:
            # 如果无法排序，随机选择
            return random.sample(population, min(count, len(population)))

    def _replace_worst(self, population: List, new_individuals: List):
        """替换最差个体"""
        if not population or not new_individuals:
            return

        try:
            # 按适应度排序，替换最差的
            population.sort(key=lambda x: getattr(x, 'fitness', 0), reverse=True)
            replace_count = min(len(new_individuals), len(population))
            population[-replace_count:] = new_individuals[:replace_count]
        except:
            # 如果无法排序，随机替换
            replace_count = min(len(new_individuals), len(population))
            indices_to_replace = random.sample(range(len(population)), replace_count)
            for idx, individual in zip(indices_to_replace, new_individuals[:replace_count]):
                population[idx] = individual

    def _calculate_diversity(self, population: List) -> float:
        """计算种群多样性"""
        if len(population) <= 1:
            return 0.0

        try:
            # 尝试基于目标空间计算多样性
            objectives = []
            for chrom in population:
                if hasattr(chrom, 'objectives'):
                    objectives.append(chrom.objectives)

            if objectives:
                obj_array = np.array(objectives)
                normalized_objs = (obj_array - obj_array.min(axis=0)) / (
                            obj_array.max(axis=0) - obj_array.min(axis=0) + 1e-10)

                total_distance = 0
                count = 0
                for i in range(len(normalized_objs)):
                    for j in range(i + 1, len(normalized_objs)):
                        distance = np.linalg.norm(normalized_objs[i] - normalized_objs[j])
                        total_distance += distance
                        count += 1

                return total_distance / count if count > 0 else 0.0
        except:
            pass

        return 0.5  # 默认中等多样性

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_rotation_statistics(self) -> Dict[str, Any]:
        """获取旋转统计"""
        if not self.rotation_history:
            return {'total_rotations': 0}

        strategies_used = {}
        for event in self.rotation_history:
            strategy = event['strategy']
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1

        return {
            'total_rotations': len(self.rotation_history),
            'strategies_used': strategies_used,
            'latest_rotation': self.rotation_history[-1] if self.rotation_history else None
        }