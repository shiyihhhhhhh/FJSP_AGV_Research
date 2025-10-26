"""
自适应NSGA-II算法 - 集成Q-Learning的自适应多目标优化
"""

import numpy as np
from typing import List, Dict, Tuple, Any
import random
from .nsga2_algorithm import NSGA2
from ..reinforcement_learning.q_learning import QLearning

class AdaptiveNSGA2(NSGA2):
    """自适应NSGA-II算法 - 集成Q-Learning的自适应优化"""

    def __init__(self, problem_data: Dict[str, Any], population_size: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 max_generations: int = 100, q_learning_config: Dict = None):

        super().__init__(problem_data, population_size, crossover_rate, mutation_rate, max_generations)

        # Q-Learning配置
        if q_learning_config is None:
            q_learning_config = {
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'exploration_rate': 0.2
            }

        # 定义状态空间
        self.state_space = self._define_state_space()

        # 定义动作空间（不同的搜索策略）
        self.action_space = self._define_action_space()

        # 初始化Q-Learning
        self.q_learning = QLearning(
            state_space=self.state_space,
            action_space=self.action_space,
            learning_rate=q_learning_config['learning_rate'],
            discount_factor=q_learning_config['discount_factor'],
            exploration_rate=q_learning_config['exploration_rate']
        )

        # 自适应参数
        self.adaptive_parameters = {
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'selection_pressure': 2.0
        }

        # 学习历史
        self.learning_history = {
            'states': [],
            'actions': [],
            'rewards': [],
            'q_values': []
        }

        # 性能跟踪
        self.performance_tracking = {
            'population_diversity': [],
            'improvement_trend': [],
            'search_stage': []
        }

    def _define_state_space(self) -> List[str]:
        """定义状态空间"""
        states = []

        # 种群多样性状态
        diversity_states = ['low_diversity', 'medium_diversity', 'high_diversity']

        # 改进趋势状态
        improvement_states = ['improving', 'stable', 'deteriorating']

        # 搜索阶段状态
        stage_states = ['early_stage', 'middle_stage', 'late_stage']

        # 帕累托前沿大小状态
        front_states = ['small_front', 'medium_front', 'large_front']

        # 组合状态
        for div in diversity_states:
            for imp in improvement_states:
                for stage in stage_states:
                    for front in front_states:
                        states.append(f"{div}_{imp}_{stage}_{front}")

        return states

    def _define_action_space(self) -> List[str]:
        """定义动作空间（不同的搜索策略）"""
        actions = [
            # 基础遗传操作组合
            'standard_pox_uniform',      # 标准POX交叉 + 均匀变异
            'enhanced_pox_swap',         # 增强POX交叉 + 交换变异
            'multi_point_crossover',     # 多点交叉
            'intensive_mutation',        # 强化变异

            # 自适应参数调整
            'increase_crossover',        # 增加交叉率
            'increase_mutation',         # 增加变异率
            'decrease_crossover',        # 减少交叉率
            'decrease_mutation',         # 减少变异率

            # 特殊搜索策略
            'diversity_preservation',    # 多样性保持
            'convergence_acceleration',  # 收敛加速
            'local_refinement',          # 局部精细化搜索
            'global_exploration'         # 全局探索
        ]
        return actions

    def _calculate_population_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population) <= 1:
            return 0.0

        # 基于目标空间的多样性
        objectives = np.array([chrom.objectives for chrom in self.population])

        # 归一化目标值
        normalized_objs = (objectives - objectives.min(axis=0)) / (objectives.max(axis=0) - objectives.min(axis=0) + 1e-10)

        # 计算平均欧氏距离
        total_distance = 0
        count = 0
        for i in range(len(normalized_objs)):
            for j in range(i + 1, len(normalized_objs)):
                distance = np.linalg.norm(normalized_objs[i] - normalized_objs[j])
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _assess_improvement_trend(self) -> str:
        """评估改进趋势"""
        if len(self.history['best_fitness']) < 5:
            return 'stable'

        recent_improvements = []
        for i in range(1, min(5, len(self.history['best_fitness']))):
            improvement = self.history['best_fitness'][i-1] - self.history['best_fitness'][i]
            recent_improvements.append(improvement)

        avg_improvement = np.mean(recent_improvements)

        if avg_improvement > 0.01:
            return 'improving'
        elif avg_improvement < -0.01:
            return 'deteriorating'
        else:
            return 'stable'

    def _determine_search_stage(self, current_gen: int) -> str:
        """确定搜索阶段"""
        progress = current_gen / self.max_generations

        if progress < 0.3:
            return 'early_stage'
        elif progress < 0.7:
            return 'middle_stage'
        else:
            return 'late_stage'

    def _get_current_state(self, current_gen: int) -> str:
        """获取当前状态"""
        # 计算种群多样性
        diversity = self._calculate_population_diversity()
        if diversity < 0.1:
            diversity_state = 'low_diversity'
        elif diversity < 0.3:
            diversity_state = 'medium_diversity'
        else:
            diversity_state = 'high_diversity'

        # 评估改进趋势
        improvement_state = self._assess_improvement_trend()

        # 确定搜索阶段
        stage_state = self._determine_search_stage(current_gen)

        # 帕累托前沿大小
        front_size = len(self.pareto_front) if self.pareto_front else 0
        if front_size < 5:
            front_state = 'small_front'
        elif front_size < 15:
            front_state = 'medium_front'
        else:
            front_state = 'large_front'

        # 组合状态
        current_state = f"{diversity_state}_{improvement_state}_{stage_state}_{front_state}"

        # 记录性能指标
        self.performance_tracking['population_diversity'].append(diversity)
        self.performance_tracking['improvement_trend'].append(improvement_state)
        self.performance_tracking['search_stage'].append(stage_state)

        return current_state

    def _calculate_reward(self, old_pareto_front: List, new_pareto_front: List) -> float:
        """计算奖励值"""
        if not old_pareto_front or not new_pareto_front:
            return 0.0

        # 计算超体积改进
        old_hv = self._calculate_hypervolume(old_pareto_front)
        new_hv = self._calculate_hypervolume(new_pareto_front)
        hv_improvement = new_hv - old_hv

        # 计算帕累托前沿大小改进
        old_size = len(old_pareto_front)
        new_size = len(new_pareto_front)
        size_improvement = (new_size - old_size) / max(old_size, 1)

        # 计算多样性改进
        old_diversity = self._calculate_population_diversity_for_front(old_pareto_front)
        new_diversity = self._calculate_population_diversity_for_front(new_pareto_front)
        diversity_improvement = new_diversity - old_diversity

        # 综合奖励
        reward = (
            0.5 * hv_improvement +      # 超体积改进权重
            0.3 * size_improvement +    # 前沿大小改进权重
            0.2 * diversity_improvement # 多样性改进权重
        )

        # 归一化奖励
        normalized_reward = max(-1.0, min(1.0, reward * 10))

        return normalized_reward

    def _calculate_population_diversity_for_front(self, front: List) -> float:
        """计算前沿多样性"""
        if len(front) <= 1:
            return 0.0

        objectives = np.array([chrom.objectives for chrom in front])
        normalized_objs = (objectives - objectives.min(axis=0)) / (objectives.max(axis=0) - objectives.min(axis=0) + 1e-10)

        total_distance = 0
        count = 0
        for i in range(len(normalized_objs)):
            for j in range(i + 1, len(normalized_objs)):
                distance = np.linalg.norm(normalized_objs[i] - normalized_objs[j])
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _apply_action(self, action: str):
        """应用选择的动作"""
        # 根据动作调整算法参数或选择操作符
        if action == 'increase_crossover':
            self.adaptive_parameters['crossover_rate'] = min(0.95, self.adaptive_parameters['crossover_rate'] + 0.1)
        elif action == 'decrease_crossover':
            self.adaptive_parameters['crossover_rate'] = max(0.1, self.adaptive_parameters['crossover_rate'] - 0.1)
        elif action == 'increase_mutation':
            self.adaptive_parameters['mutation_rate'] = min(0.5, self.adaptive_parameters['mutation_rate'] + 0.05)
        elif action == 'decrease_mutation':
            self.adaptive_parameters['mutation_rate'] = max(0.01, self.adaptive_parameters['mutation_rate'] - 0.05)
        elif action == 'diversity_preservation':
            # 多样性保持策略
            self.adaptive_parameters['crossover_rate'] = 0.7
            self.adaptive_parameters['mutation_rate'] = 0.2
        elif action == 'convergence_acceleration':
            # 收敛加速策略
            self.adaptive_parameters['crossover_rate'] = 0.9
            self.adaptive_parameters['mutation_rate'] = 0.05

        # 更新算法参数
        self.crossover_rate = self.adaptive_parameters['crossover_rate']
        self.mutation_rate = self.adaptive_parameters['mutation_rate']

    def evolve(self):
        """自适应进化一代"""
        # 记录当前帕累托前沿
        old_pareto_front = self.pareto_front.copy() if self.pareto_front else []

        # 获取当前状态
        current_gen = len(self.history['best_fitness'])
        current_state = self._get_current_state(current_gen)

        # Q-Learning选择动作
        action = self.q_learning.choose_action(current_state)

        # 应用选择的动作
        self._apply_action(action)

        # 执行标准的进化过程
        super().evolve()

        # 计算奖励
        reward = self._calculate_reward(old_pareto_front, self.pareto_front)

        # 获取新状态
        new_state = self._get_current_state(current_gen + 1)

        # 更新Q-Learning
        self.q_learning.update(current_state, action, reward, new_state)

        # 记录学习历史
        self.learning_history['states'].append(current_state)
        self.learning_history['actions'].append(action)
        self.learning_history['rewards'].append(reward)
        self.learning_history['q_values'].append(
            self.q_learning.get_q_value(current_state, action)
        )

        # 逐渐降低探索率
        if current_gen % 10 == 0:
            self.q_learning.decrease_exploration_rate()

    def run(self, max_generations: int = None):
        """运行自适应NSGA-II算法"""
        if max_generations is None:
            max_generations = self.max_generations

        print(f"🚀 开始自适应NSGA-II优化，最大代数: {max_generations}")
        print(f"🤖 Q-Learning配置: 学习率={self.q_learning.learning_rate}, "
              f"折扣因子={self.q_learning.discount_factor}, 探索率={self.q_learning.exploration_rate}")

        # 初始化种群
        self.initialize_population()

        # 初始非支配排序
        fronts = self.fast_non_dominated_sort(self.population)
        for i, front in enumerate(fronts):
            for chrom in front:
                chrom.rank = i

        # 计算初始拥挤度
        for front in fronts:
            self.crowding_distance_assignment(front)

        self._update_pareto_front()

        # 自适应进化循环
        for gen in range(max_generations):
            try:
                self.evolve()

                if (gen + 1) % 10 == 0:
                    best_makespan = min(chrom.objectives[0] for chrom in self.pareto_front) if self.pareto_front else float('inf')
                    stats = self.q_learning.get_learning_statistics()
                    print(f"  第 {gen + 1} 代 | 帕累托解: {len(self.pareto_front)} | "
                          f"最佳完工时间: {best_makespan:.2f} | "
                          f"平均奖励: {stats['average_reward']:.3f} | "
                          f"探索率: {self.q_learning.exploration_rate:.3f}")
            except Exception as e:
                print(f"⚠️ 第 {gen + 1} 代进化过程中出错: {e}")

        print(f"✅ 自适应NSGA-II优化完成，找到 {len(self.pareto_front)} 个帕累托最优解")

        # 输出学习统计
        self._print_learning_statistics()

        return self.pareto_front

    def _print_learning_statistics(self):
        """输出学习统计信息"""
        stats = self.q_learning.get_learning_statistics()
        print("\n📊 Q-Learning学习统计:")
        print(f"  总更新次数: {stats['total_updates']}")
        print(f"  平均奖励: {stats['average_reward']:.3f}")
        print(f"  最大奖励: {stats['max_reward']:.3f}")
        print(f"  最小奖励: {stats['min_reward']:.3f}")
        print(f"  访问状态数: {stats['states_visited']}")
        print(f"  使用动作数: {stats['unique_actions']}")
        print(f"  最终探索率: {self.q_learning.exploration_rate:.3f}")

        # 输出最常用的动作
        if self.learning_history['actions']:
            from collections import Counter
            action_counts = Counter(self.learning_history['actions'])
            print(f"  最常用动作: {action_counts.most_common(3)}")

    def get_learning_analysis(self) -> Dict[str, Any]:
        """获取学习分析"""
        stats = self.q_learning.get_learning_statistics()

        return {
            'q_learning_stats': stats,
            'performance_tracking': self.performance_tracking,
            'learning_history_summary': {
                'total_states': len(set(self.learning_history['states'])),
                'total_actions': len(set(self.learning_history['actions'])),
                'average_reward': np.mean(self.learning_history['rewards']) if self.learning_history['rewards'] else 0,
                'final_q_values': self.learning_history['q_values'][-10:] if len(self.learning_history['q_values']) >= 10 else self.learning_history['q_values']
            }
        }