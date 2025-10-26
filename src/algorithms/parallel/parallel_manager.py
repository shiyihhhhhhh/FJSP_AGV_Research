"""
并行计算管理类 - 负责管理多线程/多进程并行计算
"""

import multiprocessing as mp
import threading
import numpy as np
from typing import List, Dict, Any, Callable, Optional
import time
import psutil
from dataclasses import dataclass
from queue import Queue, Empty
import random


@dataclass
class ParallelConfig:
    """并行计算配置"""
    max_threads: int = 4
    max_processes: int = 2
    rotation_frequency: int = 10  # 解决方案旋转频率
    migration_rate: float = 0.1  # 迁移率
    dynamic_adjustment: bool = True  # 动态调整并行度
    cpu_threshold: float = 0.8  # CPU使用率阈值
    memory_threshold: float = 0.8  # 内存使用率阈值


class ThreadResult:
    """线程结果封装"""

    def __init__(self, thread_id: int, success: bool, data: Any = None, error: str = None):
        self.thread_id = thread_id
        self.success = success
        self.data = data
        self.error = error


class ParallelManager:
    """并行计算管理器"""

    def __init__(self, config: ParallelConfig = None):
        self.config = config or ParallelConfig()
        self.threads = []
        self.processes = []
        self.results_queue = Queue()
        self.thread_results = {}
        self.resource_monitor = ResourceMonitor()

        # 子线程状态
        self.thread_states = {}
        self.best_solutions = {}

        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0,
            'resource_utilization': []
        }

    def start_parallel_execution(self, tasks: List[Callable], task_args: List[tuple] = None) -> Dict[int, Any]:
        """启动并行执行"""
        print(f"🚀 启动并行计算，{len(tasks)} 个任务")

        if task_args is None:
            task_args = [()] * len(tasks)

        # 动态调整线程数
        optimal_threads = self._calculate_optimal_threads()
        actual_threads = min(optimal_threads, len(tasks), self.config.max_threads)

        print(f"📊 资源状况: CPU={self.resource_monitor.get_cpu_usage():.1f}%, "
              f"内存={self.resource_monitor.get_memory_usage():.1f}%")
        print(f"🔧 使用 {actual_threads} 个线程执行 {len(tasks)} 个任务")

        # 创建并启动线程
        self.threads = []
        for i in range(actual_threads):
            thread_tasks = self._distribute_tasks(tasks, task_args, i, actual_threads)
            thread = ParallelThread(
                thread_id=i,
                tasks=thread_tasks,
                results_queue=self.results_queue,
                config=self.config
            )
            self.threads.append(thread)
            thread.start()

        # 等待所有线程完成
        self._wait_for_completion()

        # 收集结果
        results = self._collect_results()

        # 更新统计信息
        self._update_statistics()

        return results

    def _calculate_optimal_threads(self) -> int:
        """计算最优线程数"""
        if not self.config.dynamic_adjustment:
            return self.config.max_threads

        cpu_usage = self.resource_monitor.get_cpu_usage() / 100.0
        memory_usage = self.resource_monitor.get_memory_usage() / 100.0

        # 基于资源使用率调整线程数
        base_threads = self.config.max_threads

        if cpu_usage > self.config.cpu_threshold or memory_usage > self.config.memory_threshold:
            # 资源紧张，减少线程数
            optimal = max(1, int(base_threads * 0.5))
        else:
            # 资源充足，可以使用更多线程
            cpu_available = 1.0 - cpu_usage
            memory_available = 1.0 - memory_usage
            availability_factor = min(cpu_available, memory_available)
            optimal = min(base_threads, int(base_threads * availability_factor * 2))

        return optimal

    def _distribute_tasks(self, tasks: List[Callable], task_args: List[tuple],
                          thread_id: int, total_threads: int) -> List[tuple]:
        """分配任务到线程"""
        thread_tasks = []
        for i in range(thread_id, len(tasks), total_threads):
            thread_tasks.append((tasks[i], task_args[i] if i < len(task_args) else ()))
        return thread_tasks

    def _wait_for_completion(self):
        """等待所有线程完成"""
        for thread in self.threads:
            thread.join()

    def _collect_results(self) -> Dict[int, Any]:
        """收集线程结果"""
        results = {}
        while not self.results_queue.empty():
            try:
                result = self.results_queue.get_nowait()
                if result.success:
                    results[result.thread_id] = result.data
                else:
                    print(f"❌ 线程 {result.thread_id} 执行失败: {result.error}")
                    self.stats['failed_tasks'] += 1
            except Empty:
                break
        return results

    def _update_statistics(self):
        """更新统计信息"""
        execution_times = []
        for thread in self.threads:
            if hasattr(thread, 'execution_time'):
                execution_times.append(thread.execution_time)

        if execution_times:
            self.stats['average_execution_time'] = np.mean(execution_times)

        # 记录资源利用率
        self.stats['resource_utilization'].append({
            'cpu': self.resource_monitor.get_cpu_usage(),
            'memory': self.resource_monitor.get_memory_usage(),
            'timestamp': time.time()
        })

    def perform_solution_rotation(self, subpopulations: List[List], rotation_strategy: str = "elite") -> List[List]:
        """执行解决方案旋转"""
        print(f"🔄 执行解决方案旋转，策略: {rotation_strategy}")

        if len(subpopulations) <= 1:
            return subpopulations

        rotated_populations = [pop.copy() for pop in subpopulations]

        if rotation_strategy == "elite":
            # 精英迁移策略
            self._elite_migration(rotated_populations)
        elif rotation_strategy == "random":
            # 随机迁移策略
            self._random_migration(rotated_populations)
        elif rotation_strategy == "ring":
            # 环状迁移策略
            self._ring_migration(rotated_populations)
        elif rotation_strategy == "fully_connected":
            # 全连接迁移策略
            self._fully_connected_migration(rotated_populations)

        return rotated_populations

    def _elite_migration(self, populations: List[List]):
        """精英迁移策略"""
        n_populations = len(populations)
        migration_count = max(1, int(len(populations[0]) * self.config.migration_rate))

        for i in range(n_populations):
            # 选择精英个体
            source_idx = i
            target_idx = (i + 1) % n_populations

            # 从源种群选择精英
            if len(populations[source_idx]) > migration_count:
                elites = sorted(populations[source_idx],
                                key=lambda x: getattr(x, 'fitness', 0))[:migration_count]

                # 迁移到目标种群，替换最差个体
                if len(populations[target_idx]) >= migration_count:
                    # 按适应度排序，替换最差的
                    populations[target_idx].sort(key=lambda x: getattr(x, 'fitness', 0), reverse=True)
                    populations[target_idx][-migration_count:] = elites

        print(f"✅ 精英迁移完成，每个种群迁移 {migration_count} 个个体")

    def _random_migration(self, populations: List[List]):
        """随机迁移策略"""
        n_populations = len(populations)
        migration_count = max(1, int(len(populations[0]) * self.config.migration_rate))

        for i in range(n_populations):
            source_idx = i
            target_idx = (i + 1) % n_populations

            # 随机选择个体迁移
            if len(populations[source_idx]) > migration_count:
                migrants = random.sample(populations[source_idx], migration_count)

                # 随机替换目标种群的个体
                if len(populations[target_idx]) >= migration_count:
                    indices_to_replace = random.sample(range(len(populations[target_idx])), migration_count)
                    for idx, migrant in zip(indices_to_replace, migrants):
                        populations[target_idx][idx] = migrant

        print(f"✅ 随机迁移完成，每个种群迁移 {migration_count} 个个体")

    def _ring_migration(self, populations: List[List]):
        """环状迁移策略"""
        n_populations = len(populations)
        migration_count = max(1, int(len(populations[0]) * self.config.migration_rate))

        # 创建环状拓扑
        for i in range(n_populations):
            source = populations[i]
            target = populations[(i + 1) % n_populations]

            if len(source) > migration_count and len(target) >= migration_count:
                # 选择源种群的精英
                elites = sorted(source, key=lambda x: getattr(x, 'fitness', 0))[:migration_count]

                # 替换目标种群的最差个体
                target.sort(key=lambda x: getattr(x, 'fitness', 0), reverse=True)
                target[-migration_count:] = elites

        print(f"✅ 环状迁移完成")

    def _fully_connected_migration(self, populations: List[List]):
        """全连接迁移策略"""
        n_populations = len(populations)
        migration_count = max(1, int(len(populations[0]) * self.config.migration_rate * 0.5))

        # 每个种群向所有其他种群迁移
        for i in range(n_populations):
            for j in range(n_populations):
                if i != j:
                    if (len(populations[i]) > migration_count and
                            len(populations[j]) >= migration_count):
                        # 选择精英个体
                        elites = sorted(populations[i],
                                        key=lambda x: getattr(x, 'fitness', 0))[:migration_count]

                        # 替换目标种群的最差个体
                        populations[j].sort(key=lambda x: getattr(x, 'fitness', 0), reverse=True)
                        populations[j][-migration_count:] = elites

        print(f"✅ 全连接迁移完成")

    def get_performance_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'total_threads_used': len(self.threads),
            'total_tasks': self.stats['total_tasks'],
            'completed_tasks': self.stats['completed_tasks'],
            'failed_tasks': self.stats['failed_tasks'],
            'success_rate': (self.stats['completed_tasks'] / self.stats['total_tasks']
                             if self.stats['total_tasks'] > 0 else 0),
            'average_execution_time': self.stats['average_execution_time'],
            'current_cpu_usage': self.resource_monitor.get_cpu_usage(),
            'current_memory_usage': self.resource_monitor.get_memory_usage()
        }


class ParallelThread(threading.Thread):
    """并行线程类"""

    def __init__(self, thread_id: int, tasks: List[tuple], results_queue: Queue, config: ParallelConfig):
        super().__init__()
        self.thread_id = thread_id
        self.tasks = tasks
        self.results_queue = results_queue
        self.config = config
        self.execution_time = 0
        self.completed_tasks = 0

    def run(self):
        """线程执行"""
        start_time = time.time()

        for task, args in self.tasks:
            try:
                # 执行任务
                result = task(*args)
                self.results_queue.put(ThreadResult(
                    thread_id=self.thread_id,
                    success=True,
                    data=result
                ))
                self.completed_tasks += 1
            except Exception as e:
                self.results_queue.put(ThreadResult(
                    thread_id=self.thread_id,
                    success=False,
                    error=str(e)
                ))

        self.execution_time = time.time() - start_time


class ResourceMonitor:
    """资源监控器"""

    def __init__(self):
        self.cpu_usage = 0
        self.memory_usage = 0
        self.update_interval = 2  # 更新间隔（秒）
        self.last_update = 0

    def get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        self._update_if_needed()
        return self.cpu_usage

    def get_memory_usage(self) -> float:
        """获取内存使用率"""
        self._update_if_needed()
        return self.memory_usage

    def _update_if_needed(self):
        """如果需要则更新资源信息"""
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            self.cpu_usage = psutil.cpu_percent(interval=None)
            self.memory_usage = psutil.virtual_memory().percent
            self.last_update = current_time