from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np


class Chromosome:
    """染色体类 - 两层编码结构，与您的模型兼容"""

    def __init__(self, problem_data: Dict[str, Any]):
        self.problem_data = problem_data
        self.OS = []  # 工序排序向量 Operation Sequence
        self.MA = []  # 机器&AGV分配矩阵 Machine-AGV Assignment
        self.fitness = None  # 适应度值 (多个目标值)
        self.schedule = None  # 解码后的调度方案
        self.objectives = None  # 目标函数值 (makespan, energy, cost)

    def initialize(self, method: str = "random"):
        """初始化染色体

        Args:
            method: 初始化方法 ['random', 'neh', 'spt', 'lpt']
        """
        if method == "random":
            self._initialize_random()
        elif method == "neh":
            self._initialize_neh()
        elif method == "spt":
            self._initialize_spt()
        elif method == "lpt":
            self._initialize_lpt()
        else:
            raise ValueError(f"未知的初始化方法: {method}")

    def _initialize_random(self):
        """随机初始化"""
        n_jobs = self.problem_data['n_jobs']
        n_operations = self.problem_data['n_operations']

        # 初始化OS向量
        self.OS = []
        for job_id in range(n_jobs):
            for op_id in range(n_operations[job_id]):
                self.OS.append(job_id)
        np.random.shuffle(self.OS)

        # 初始化MA矩阵
        self.MA = []
        for job_id in range(n_jobs):
            job_assignments = []
            for op_id in range(n_operations[job_id]):
                # 随机选择可用机器
                available_machines = self._get_available_machines(job_id, op_id)
                machine_id = np.random.choice(available_machines)

                # 随机选择AGV
                agv_id = np.random.randint(0, self.problem_data['n_agvs'])

                job_assignments.append((machine_id, agv_id))

            self.MA.append(job_assignments)

    def _initialize_neh(self):
        """NEH启发式初始化"""
        n_jobs = self.problem_data['n_jobs']
        n_operations = self.problem_data['n_operations']

        # 计算每个工件的总加工时间（选择最短加工时间）
        job_times = []
        for job_id in range(n_jobs):
            total_time = 0
            for op_id in range(n_operations[job_id]):
                min_time = min(
                    self.problem_data['processing_time'].get((job_id, op_id, machine), float('inf'))
                    for machine in range(self.problem_data['n_machines'])
                )
                if min_time < float('inf'):
                    total_time += min_time
            job_times.append((job_id, total_time))

        # 按总加工时间降序排序
        job_times.sort(key=lambda x: x[1], reverse=True)

        # 构建OS序列：按顺序展开工件
        self.OS = []
        for job_id, _ in job_times:
            for _ in range(n_operations[job_id]):
                self.OS.append(job_id)

        # 随机初始化MA矩阵
        self._initialize_random_ma()

    def _initialize_spt(self):
        """最短加工时间优先初始化"""
        self._initialize_priority_based('spt')

    def _initialize_lpt(self):
        """最长加工时间优先初始化"""
        self._initialize_priority_based('lpt')

    def _initialize_priority_based(self, priority_type: str):
        """基于优先级的初始化"""
        n_jobs = self.problem_data['n_jobs']
        n_operations = self.problem_data['n_operations']

        operation_priorities = []

        for job_id in range(n_jobs):
            for op_id in range(n_operations[job_id]):
                # 计算平均加工时间
                times = [
                    self.problem_data['processing_time'].get((job_id, op_id, machine), float('inf'))
                    for machine in range(self.problem_data['n_machines'])
                ]
                valid_times = [t for t in times if t < float('inf')]
                if not valid_times:
                    avg_time = 0
                else:
                    avg_time = np.mean(valid_times)

                if priority_type == 'spt':
                    priority = -avg_time  # 时间越短优先级越高
                else:  # lpt
                    priority = avg_time  # 时间越长优先级越高

                operation_priorities.append((job_id, op_id, priority))

        # 按优先级排序
        operation_priorities.sort(key=lambda x: x[2])

        # 构建OS序列
        self.OS = [job_id for job_id, _, _ in operation_priorities]

        # 随机初始化MA矩阵
        self._initialize_random_ma()

    def _initialize_random_ma(self):
        """随机初始化MA矩阵"""
        n_jobs = self.problem_data['n_jobs']
        n_operations = self.problem_data['n_operations']

        self.MA = []
        for job_id in range(n_jobs):
            job_assignments = []
            for op_id in range(n_operations[job_id]):
                available_machines = self._get_available_machines(job_id, op_id)
                machine_id = np.random.choice(available_machines)
                agv_id = np.random.randint(0, self.problem_data['n_agvs'])
                job_assignments.append((machine_id, agv_id))
            self.MA.append(job_assignments)

    def _get_available_machines(self, job_id: int, op_id: int) -> List[int]:
        """获取可用机器列表"""
        available_machines = []
        for machine_id in range(self.problem_data['n_machines']):
            if (job_id, op_id, machine_id) in self.problem_data['processing_time']:
                available_machines.append(machine_id)

        # 如果没有可用机器，返回所有机器
        if not available_machines:
            available_machines = list(range(self.problem_data['n_machines']))

        return available_machines

    def copy(self):
        """深拷贝染色体"""
        new_chromosome = Chromosome(self.problem_data)
        new_chromosome.OS = self.OS.copy()
        new_chromosome.MA = [row.copy() for row in self.MA]
        new_chromosome.fitness = self.fitness
        new_chromosome.schedule = self.schedule
        new_chromosome.objectives = self.objectives
        return new_chromosome

    def __str__(self):
        return f"Chromosome(OS={self.OS}, Objectives={self.objectives})"


class BaseMetaheuristic(ABC):
    """元启发式算法基类"""

    def __init__(self, problem_data: Dict[str, Any], population_size: int = 100):
        self.problem_data = problem_data
        self.population_size = population_size
        self.population = []
        self.best_solution = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': []
        }

    @abstractmethod
    def initialize_population(self):
        """初始化种群"""
        pass

    @abstractmethod
    def evolve(self):
        """进化一代"""
        pass

    @abstractmethod
    def run(self, max_generations: int):
        """运行算法"""
        pass