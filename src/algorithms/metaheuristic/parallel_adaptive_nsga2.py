"""
并行自适应NSGA-II算法 - 集成并行计算的自适应多目标优化
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import random
import time
from .adaptive_nsga2 import AdaptiveNSGA2
from ..parallel.parallel_manager import ParallelManager, ParallelConfig
from ..parallel.solution_rotator import SolutionRotator


class ParallelAdaptiveNSGA2(AdaptiveNSGA2):
    """并行自适应NSGA-II算法"""

    def __init__(self, problem_data: Dict[str, Any], population_size: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 max_generations: int = 100, q_learning_config: Dict = None,
                 parallel_config: ParallelConfig = None):

        super().__init__(problem_data, population_size, crossover_rate,
                         mutation_rate, max_generations, q_learning_config)

        # 并行计算配置
        self.parallel_config = parallel_config or ParallelConfig()
        self.parallel_manager = ParallelManager(self.parallel_config)
        self.solution_rotator = SolutionRotator()

        # 子种群管理
        self.subpopulations = []
        self.subpopulation_best = []
        self.migration_history = []

        # 并行统计
        self.parallel_stats = {
            'subpopulation_sizes': [],
            'migration_events': [],
            'resource_utilization': [],
            'speedup_factor': 1.0
        }

    def initialize_parallel_population(self):
        """初始化并行种群"""
        print("🧬 初始化并行自适应NSGA-II种群...")

        # 计算子种群数量
        n_subpopulations = self._calculate_optimal_subpopulations()
        subpopulation_size = max(10, self.population_size // n_subpopulations)

        print(f"📊 创建 {n_subpopulations} 个子种群，每个大小约 {subpopulation_size}")

        # 创建子种群
        self.subpopulations = []
        tasks = []

        for i in range(n_subpopulations):
            # 为每个子种群创建初始化任务
            task = (self._initialize_subpopulation, (subpopulation_size, i))
            tasks.append(task)

        # 并行执行初始化
        results = self.parallel_manager.start_parallel_execution(
            [task[0] for task in tasks],
            [task[1] for task in tasks]
        )

        # 收集子种群
        for thread_id, subpopulation in results.items():
            if subpopulation:
                self.subpopulations.append(subpopulation)

        # 如果子种群数量不足，补充创建
        while len(self.subpopulations) < n_subpopulations:
            subpop = self._initialize_subpopulation(subpopulation_size, len(self.subpopulations))
            self.subpopulations.append(subpop)

        # 更新主种群
        self.population = []
        for subpop in self.subpopulations:
            self.population.extend(subpop)

        # 确保总种群大小正确
        if len(self.population) > self.population_size:
            self.population = self.population[:self.population_size]
        elif len(self.population) < self.population_size:
            # 补充个体
            additional = self.population_size - len(self.population)
            for _ in range(additional):
                chrom = self._create_random_chromosome()
                self.population.append(chrom)

        print(f"✅ 并行种群初始化完成，总大小: {len(self.population)}")
        self.parallel_stats['subpopulation_sizes'] = [len(subpop) for subpop in self.subpopulations]

    def _calculate_optimal_subpopulations(self) -> int:
        """计算最优子种群数量"""
        # 基于问题规模和可用资源
        base_subpopulations = min(8, max(2, self.population_size // 50))

        if self.parallel_config.dynamic_adjustment:
            # 基于资源使用率调整
            cpu_usage = self.parallel_manager.resource_monitor.get_cpu_usage() / 100.0
            if cpu_usage > self.parallel_config.cpu_threshold:
                return max(2, base_subpopulations // 2)
            else:
                return min(self.parallel_config.max_threads, base_subpopulations)
        else:
            return min(self.parallel_config.max_threads, base_subpopulations)

    def _initialize_subpopulation(self, size: int, subpop_id: int) -> List:
        """初始化子种群"""
        subpopulation = []
        methods = ['random', 'neh', 'spt', 'lpt']

        for i in range(size):
            method = random.choice(methods)
            chromosome = self._create_chromosome(method)
            subpopulation.append(chromosome)

        return subpopulation

    def _create_chromosome(self, method: str):
        """创建染色体"""
        chromosome = Chromosome(self.problem_data)
        chromosome.initialize(method)
        self.decoder.decode(chromosome)
        return chromosome

    def _create_random_chromosome(self):
        """创建随机染色体"""
        return self._create_chromosome('random')

    def parallel_evolve(self):
        """并行进化一代"""
        # 记录当前帕累托前沿
        old_pareto_front = self.pareto_front.copy() if self.pareto_front else []

        # 获取当前状态
        current_gen = len(self.history['best_fitness'])
        current_state = self._get_current_state(current_gen)

        # Q-Learning选择动作
        action = self.q_learning.choose_action(current_state)

        # 应用选择的动作
        self._apply_action(action)

        # 并行执行子种群进化
        self._parallel_subpopulation_evolution()

        # 定期执行解决方案旋转
        if current_gen % self.parallel_config.rotation_frequency == 0:
            self._perform_solution_rotation()

        # 合并子种群更新主种群
        self._merge_subpopulations()

        # 更新帕累托前沿
        self._update_pareto_front()

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

    def _parallel_subpopulation_evolution(self):
        """并行执行子种群进化"""
        tasks = []

        for i, subpopulation in enumerate(self.subpopulations):
            task = (self._evolve_subpopulation, (subpopulation.copy(), i))
            tasks.append(task)

        # 并行执行进化
        results = self.parallel_manager.start_parallel_execution(
            [task[0] for task in tasks],
            [task[1] for task in tasks]
        )

        # 更新子种群
        for thread_id, evolved_subpop in results.items():
            if evolved_subpop and thread_id < len(self.subpopulations):
                self.subpopulations[thread_id] = evolved_subpop

    def _evolve_subpopulation(self, subpopulation: List, subpop_id: int) -> List:
        """进化子种群（在并行线程中执行）"""
        if len(subpopulation) <= 1:
            return subpopulation

        # 创建子种群的临时算法实例
        temp_nsga2 = AdaptiveNSGA2(
            self.problem_data,
            population_size=len(subpopulation),
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            max_generations=1
        )

        # 设置子种群
        temp_nsga2.population = subpopulation

        # 执行一代进化
        try:
            temp_nsga2.evolve()
            return temp_nsga2.population
        except Exception as e:
            print(f"⚠️ 子种群 {subpop_id} 进化失败: {e}")
            return subpopulation

    def _perform_solution_rotation(self):
        """执行解决方案旋转"""
        print("🔄 执行并行解决方案旋转...")

        # 使用并行管理器进行解决方案旋转
        rotated_populations = self.parallel_manager.perform_solution_rotation(
            self.subpopulations,
            rotation_strategy="elite"
        )

        self.subpopulations = rotated_populations

        # 记录迁移历史
        self.migration_history.append({
            'generation': len(self.history['best_fitness']),
            'subpopulations': len(self.subpopulations),
            'migration_strategy': 'elite'
        })

        self.parallel_stats['migration_events'].append(len(self.migration_history))

    def _merge_subpopulations(self):
        """合并子种群到主种群"""
        self.population = []
        for subpop in self.subpopulations:
            self.population.extend(subpop)

        # 确保总种群大小正确
        if len(self.population) > self.population_size:
            # 随机选择保留的个体
            self.population = random.sample(self.population, self.population_size)
        elif len(self.population) < self.population_size:
            # 补充个体
            additional = self.population_size - len(self.population)
            for _ in range(additional):
                chrom = self._create_random_chromosome()
                self.population.append(chrom)

    def run(self, max_generations: int = None):
        """运行并行自适应NSGA-II算法"""
        if max_generations is None:
            max_generations = self.max_generations

        print(f"🚀 启动并行自适应NSGA-II优化，最大代数: {max_generations}")
        print(f"🤖 Q-Learning配置: 学习率={self.q_learning.learning_rate}, "
              f"折扣因子={self.q_learning.discount_factor}, 探索率={self.q_learning.exploration_rate}")
        print(f"🔧 并行配置: 最大线程={self.parallel_config.max_threads}, "
              f"旋转频率={self.parallel_config.rotation_frequency}")

        start_time = time.time()

        # 初始化并行种群
        self.initialize_parallel_population()

        # 初始非支配排序
        fronts = self.fast_non_dominated_sort(self.population)
        for i, front in enumerate(fronts):
            for chrom in front:
                chrom.rank = i

        # 计算初始拥挤度
        for front in fronts:
            self.crowding_distance_assignment(front)

        self._update_pareto_front()

        # 并行进化循环
        for gen in range(max_generations):
            try:
                self.parallel_evolve()

                if (gen + 1) % 10 == 0:
                    best_makespan = min(
                        chrom.objectives[0] for chrom in self.pareto_front) if self.pareto_front else float('inf')
                    stats = self.q_learning.get_learning_statistics()
                    parallel_stats = self.parallel_manager.get_performance_statistics()

                    print(f"  第 {gen + 1} 代 | 帕累托解: {len(self.pareto_front)} | "
                          f"最佳完工时间: {best_makespan:.2f} | "
                          f"平均奖励: {stats['average_reward']:.3f} | "
                          f"子种群: {len(self.subpopulations)} | "
                          f"CPU使用: {parallel_stats['current_cpu_usage']:.1f}%")

            except Exception as e:
                print(f"⚠️ 第 {gen + 1} 代并行进化过程中出错: {e}")

        total_time = time.time() - start_time

        print(f"✅ 并行自适应NSGA-II优化完成，找到 {len(self.pareto_front)} 个帕累托最优解")
        print(f"⏱️ 总执行时间: {total_time:.2f} 秒")

        # 输出学习和并行统计
        self._print_learning_statistics()
        self._print_parallel_statistics(total_time)

        return self.pareto_front

    def _print_parallel_statistics(self, total_time: float):
        """输出并行统计信息"""
        parallel_stats = self.parallel_manager.get_performance_statistics()

        print("\n📊 并行计算统计:")
        print(f"  使用线程数: {parallel_stats['total_threads_used']}")
        print(f"  任务成功率: {parallel_stats['success_rate']:.1%}")
        print(f"  平均执行时间: {parallel_stats['average_execution_time']:.3f} 秒")
        print(f"  子种群数量: {len(self.subpopulations)}")
        print(f"  迁移事件: {len(self.migration_history)} 次")
        print(f"  最终CPU使用率: {parallel_stats['current_cpu_usage']:.1f}%")
        print(f"  最终内存使用率: {parallel_stats['current_memory_usage']:.1f}%")

        # 计算加速比
        if parallel_stats['average_execution_time'] > 0:
            estimated_sequential_time = parallel_stats['average_execution_time'] * parallel_stats['total_threads_used']
            if estimated_sequential_time > 0:
                speedup = estimated_sequential_time / total_time
                print(f"  估计加速比: {speedup:.2f}x")

    def get_parallel_analysis(self) -> Dict[str, Any]:
        """获取并行分析"""
        learning_analysis = super().get_learning_analysis()
        parallel_stats = self.parallel_manager.get_performance_statistics()

        return {
            **learning_analysis,
            'parallel_stats': parallel_stats,
            'subpopulation_info': {
                'total_subpopulations': len(self.subpopulations),
                'subpopulation_sizes': [len(subpop) for subpop in self.subpopulations],
                'migration_history': self.migration_history
            },
            'resource_utilization': self.parallel_stats['resource_utilization']
        }