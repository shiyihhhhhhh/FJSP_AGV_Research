import numpy as np
from typing import List, Dict, Tuple, Any
import random

# 使用相对导入
from .base_metaheuristic import BaseMetaheuristic, Chromosome
from .nsga2 import ScheduleDecoder


class NSGA2(BaseMetaheuristic):
    """NSGA-II多目标优化算法"""

    def __init__(self, problem_data: Dict[str, Any], population_size: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 max_generations: int = 100):
        super().__init__(problem_data, population_size)

        # 算法参数
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations

        # 解码器
        self.decoder = ScheduleDecoder(problem_data)

        # 帕累托前沿
        self.pareto_front = []
        self.history['pareto_front_sizes'] = []
        self.history['hypervolume'] = []

    def initialize_population(self):
        """初始化种群"""
        print("🧬 初始化NSGA-II种群...")
        self.population = []

        # 使用多种初始化方法创建多样化的初始种群
        methods = ['random', 'neh', 'spt', 'lpt']
        method_weights = [0.4, 0.3, 0.2, 0.1]

        for i in range(self.population_size):
            method = np.random.choice(methods, p=method_weights)
            chromosome = Chromosome(self.problem_data)
            chromosome.initialize(method)

            # 解码并计算目标值
            self.decoder.decode(chromosome)
            self.population.append(chromosome)

        print(f"✅ 种群初始化完成，大小: {len(self.population)}")

    def fast_non_dominated_sort(self, population: List[Chromosome]) -> List[List[Chromosome]]:
        """快速非支配排序"""
        fronts = [[]]

        # 为每个个体计算支配关系
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []

            for q in population:
                if self._dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self._dominates(q, p):
                    p.domination_count += 1

            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _dominates(self, chrom1: Chromosome, chrom2: Chromosome) -> bool:
        """判断chrom1是否支配chrom2"""
        obj1 = chrom1.objectives
        obj2 = chrom2.objectives

        # 所有目标都不差，且至少一个目标更好
        not_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))

        return not_worse and better

    def crowding_distance_assignment(self, front: List[Chromosome]) -> None:
        """计算拥挤度"""
        if not front:
            return

        num_objectives = len(front[0].objectives)

        for chrom in front:
            chrom.crowding_distance = 0

        for obj_idx in range(num_objectives):
            # 按当前目标函数值排序
            front.sort(key=lambda x: x.objectives[obj_idx])

            # 边界个体的拥挤度为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # 计算中间个体的拥挤度
            min_obj = front[0].objectives[obj_idx]
            max_obj = front[-1].objectives[obj_idx]

            if max_obj - min_obj < 1e-10:
                continue

            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                                                      front[i + 1].objectives[obj_idx] - front[i - 1].objectives[
                                                  obj_idx]
                                              ) / (max_obj - min_obj)

    def selection(self, population: List[Chromosome]) -> List[Chromosome]:
        """锦标赛选择"""
        selected = []

        while len(selected) < self.population_size:
            # 随机选择两个个体
            candidates = random.sample(population, 2)

            # 确保两个个体都有拥挤度属性
            for candidate in candidates:
                if not hasattr(candidate, 'crowding_distance'):
                    candidate.crowding_distance = 0.0
                if not hasattr(candidate, 'rank'):
                    candidate.rank = 0

            # 选择较优的个体
            if self._is_better(candidates[0], candidates[1]):
                selected.append(candidates[0].copy())
            else:
                selected.append(candidates[1].copy())

        return selected

    def _is_better(self, chrom1: Chromosome, chrom2: Chromosome) -> bool:
        """比较两个个体的优劣"""
        # 确保两个个体都有必要的属性
        if not hasattr(chrom1, 'rank'):
            chrom1.rank = 0
        if not hasattr(chrom2, 'rank'):
            chrom2.rank = 0
        if not hasattr(chrom1, 'crowding_distance'):
            chrom1.crowding_distance = 0.0
        if not hasattr(chrom2, 'crowding_distance'):
            chrom2.crowding_distance = 0.0

        if chrom1.rank < chrom2.rank:
            return True
        elif chrom1.rank == chrom2.rank:
            return chrom1.crowding_distance > chrom2.crowding_distance
        else:
            return False

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """交叉操作"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        if random.random() < self.crossover_rate:
            # OS向量的交叉（改进的POX交叉）
            self._improved_pox_crossover(parent1, parent2, child1, child2)

            # MA矩阵的交叉（均匀交叉）
            self._uniform_ma_crossover(parent1, parent2, child1, child2)

        return child1, child2

    def _improved_pox_crossover(self, parent1: Chromosome, parent2: Chromosome,
                                child1: Chromosome, child2: Chromosome):
        """改进的工序优先交叉 - 修复索引越界问题"""
        n_jobs = self.problem_data['n_jobs']

        # 确保job_subset不是空集也不是全集
        if n_jobs <= 1:
            # 如果只有一个工件，直接交换
            child1.OS, child2.OS = parent2.OS.copy(), parent1.OS.copy()
            return

        job_subset = set(random.sample(range(n_jobs), random.randint(1, n_jobs - 1)))

        # 构建子代OS - 使用更安全的方法
        child1_os = []
        child2_os = []

        # 对于child1: 从parent1保留job_subset中的工序顺序，从parent2获取其他工序
        for job in parent1.OS:
            if job in job_subset:
                child1_os.append(job)

        for job in parent2.OS:
            if job not in job_subset:
                child1_os.append(job)

        # 对于child2: 从parent2保留job_subset中的工序顺序，从parent1获取其他工序
        for job in parent2.OS:
            if job in job_subset:
                child2_os.append(job)

        for job in parent1.OS:
            if job not in job_subset:
                child2_os.append(job)

        # 验证长度是否正确
        if len(child1_os) != len(parent1.OS):
            # 使用父代作为备选
            child1_os = parent1.OS.copy()

        if len(child2_os) != len(parent2.OS):
            child2_os = parent2.OS.copy()

        child1.OS = child1_os
        child2.OS = child2_os

    def _uniform_ma_crossover(self, parent1: Chromosome, parent2: Chromosome,
                              child1: Chromosome, child2: Chromosome):
        """MA矩阵的均匀交叉"""
        for job_id in range(len(parent1.MA)):
            for op_id in range(len(parent1.MA[job_id])):
                if random.random() < 0.5:
                    # 确保索引有效
                    if (job_id < len(child2.MA) and op_id < len(child2.MA[job_id]) and
                            job_id < len(child1.MA) and op_id < len(child1.MA[job_id])):
                        child1.MA[job_id][op_id], child2.MA[job_id][op_id] = \
                            child2.MA[job_id][op_id], child1.MA[job_id][op_id]

    def mutation(self, chromosome: Chromosome):
        """变异操作"""
        if random.random() < self.mutation_rate:
            self._swap_mutation_os(chromosome)
            self._reset_mutation_ma(chromosome)

    def _swap_mutation_os(self, chromosome: Chromosome):
        """OS向量的交换变异"""
        if len(chromosome.OS) < 2:
            return

        idx1, idx2 = random.sample(range(len(chromosome.OS)), 2)
        chromosome.OS[idx1], chromosome.OS[idx2] = chromosome.OS[idx2], chromosome.OS[idx1]

    def _reset_mutation_ma(self, chromosome: Chromosome):
        """MA矩阵的随机重置变异"""
        if not chromosome.MA:
            return

        job_id = random.randint(0, len(chromosome.MA) - 1)
        if chromosome.MA[job_id]:
            op_id = random.randint(0, len(chromosome.MA[job_id]) - 1)

            available_machines = self._get_available_machines(job_id, op_id)
            if available_machines:
                new_machine = random.choice(available_machines)
                new_agv = random.randint(0, self.problem_data['n_agvs'] - 1)
                chromosome.MA[job_id][op_id] = (new_machine, new_agv)

    def _get_available_machines(self, job_id: int, op_id: int) -> List[int]:
        """获取可用机器列表"""
        available_machines = []
        for machine_id in range(self.problem_data['n_machines']):
            if (job_id, op_id, machine_id) in self.problem_data['processing_time']:
                available_machines.append(machine_id)

        if not available_machines:
            available_machines = list(range(self.problem_data['n_machines']))

        return available_machines

    def evolve(self):
        """进化一代"""
        # 1. 生成子代
        offspring = []
        parents = self.selection(self.population)

        # 交叉和变异生成子代
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                try:
                    child1, child2 = self.crossover(parents[i], parents[i + 1])
                    self.mutation(child1)
                    self.mutation(child2)

                    self.decoder.decode(child1)
                    self.decoder.decode(child2)

                    offspring.extend([child1, child2])
                except Exception as e:
                    # 如果出错，使用父代作为备选
                    offspring.extend([parents[i].copy(), parents[i + 1].copy()])

        # 2. 合并父代和子代
        combined_population = self.population + offspring

        # 3. 快速非支配排序
        fronts = self.fast_non_dominated_sort(combined_population)

        # 确保所有个体都有rank属性
        for i, front in enumerate(fronts):
            for chrom in front:
                chrom.rank = i

        # 4. 计算所有前沿的拥挤度
        for front in fronts:
            self.crowding_distance_assignment(front)

        # 5. 构建新种群
        new_population = []
        front_idx = 0

        while front_idx < len(fronts) and len(new_population) + len(fronts[front_idx]) <= self.population_size:
            # 确保前沿中的个体都有拥挤度属性
            for chrom in fronts[front_idx]:
                if not hasattr(chrom, 'crowding_distance'):
                    chrom.crowding_distance = 0.0
            new_population.extend(fronts[front_idx])
            front_idx += 1

        # 6. 如果还需要更多个体，按拥挤度选择
        if len(new_population) < self.population_size and front_idx < len(fronts):
            last_front = fronts[front_idx]
            # 确保所有个体都有拥挤度属性
            for chrom in last_front:
                if not hasattr(chrom, 'crowding_distance'):
                    chrom.crowding_distance = 0.0
            self.crowding_distance_assignment(last_front)
            last_front.sort(key=lambda x: x.crowding_distance, reverse=True)

            needed = self.population_size - len(new_population)
            new_population.extend(last_front[:needed])

        self.population = new_population

        # 7. 更新帕累托前沿
        self._update_pareto_front()

    def _update_pareto_front(self):
        """更新帕累托前沿"""
        try:
            current_front = self.fast_non_dominated_sort(self.population)
            if current_front:
                self.pareto_front = current_front[0]  # 第一个前沿就是帕累托前沿
                self.history['pareto_front_sizes'].append(len(current_front[0]))

                if current_front[0]:
                    hv = self._calculate_hypervolume(current_front[0])
                    self.history['hypervolume'].append(hv)
        except Exception as e:
            print(f"⚠️ 更新帕累托前沿时出错: {e}")

    def _calculate_hypervolume(self, front: List[Chromosome]) -> float:
        """计算超体积指标（简化版本）"""
        if not front:
            return 0.0

        try:
            ref_point = [max(chrom.objectives[i] for chrom in front) for i in range(3)]
            volume = 1.0
            for obj_idx in range(3):
                min_obj = min(chrom.objectives[obj_idx] for chrom in front)
                volume *= (ref_point[obj_idx] - min_obj + 1)

            return volume
        except:
            return 0.0

    def run(self, max_generations: int = None):
        """运行NSGA-II算法"""
        if max_generations is None:
            max_generations = self.max_generations

        print(f"🚀 开始NSGA-II优化，最大代数: {max_generations}")

        # 初始化种群
        self.initialize_population()

        # 初始非支配排序
        try:
            fronts = self.fast_non_dominated_sort(self.population)
            for i, front in enumerate(fronts):
                for chrom in front:
                    chrom.rank = i

            # 计算初始拥挤度
            for front in fronts:
                self.crowding_distance_assignment(front)

            self._update_pareto_front()
        except Exception as e:
            print(f"⚠️ 初始排序时出错: {e}")

        # 进化循环
        for gen in range(max_generations):
            try:
                self.evolve()

                if (gen + 1) % 10 == 0:
                    if self.pareto_front:
                        best_makespan = min(chrom.objectives[0] for chrom in self.pareto_front)
                        print(f"  第 {gen + 1} 代 | 帕累托解: {len(self.pareto_front)} | "
                              f"最佳完工时间: {best_makespan:.2f}")
                    else:
                        print(f"  第 {gen + 1} 代 | 暂无帕累托解")
            except Exception as e:
                print(f"⚠️ 第 {gen + 1} 代进化过程中出错: {e}")

        print(f"✅ NSGA-II优化完成，找到 {len(self.pareto_front)} 个帕累托最优解")

        return self.pareto_front

    def get_best_solutions(self, objective_weights: List[float] = None):
        """根据权重获取最佳解"""
        if not self.pareto_front:
            return []

        if objective_weights is None:
            objective_weights = [1 / 3, 1 / 3, 1 / 3]

        try:
            scored_solutions = []
            for chrom in self.pareto_front:
                score = sum(w * obj for w, obj in zip(objective_weights, chrom.objectives))
                scored_solutions.append((score, chrom))

            scored_solutions.sort(key=lambda x: x[0])

            return [chrom for _, chrom in scored_solutions]
        except:
            return self.pareto_front