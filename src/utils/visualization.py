"""
可视化工具
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional


class GanttChart:
    """甘特图绘制类"""

    @staticmethod
    def plot_machine_gantt(schedule: Dict, save_path: Optional[str] = None):
        """绘制机器调度甘特图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.Set3(np.linspace(0, 1, 10))

        machine_tasks = {}
        for key, task_info in schedule['job_schedule'].items():
            machine_id = task_info['machine']
            if machine_id not in machine_tasks:
                machine_tasks[machine_id] = []

            machine_tasks[machine_id].append({
                'job': key[0],
                'operation': key[1],
                'start': task_info['process'][0],
                'end': task_info['process'][1],
                'duration': task_info['process'][1] - task_info['process'][0]
            })

        # 绘制每个机器的任务
        for machine_id, tasks in machine_tasks.items():
            for task in tasks:
                ax.barh(machine_id, task['duration'],
                        left=task['start'], height=0.6,
                        color=colors[task['job'] % len(colors)],
                        edgecolor='black', alpha=0.7)

                # 添加文本标签
                ax.text(task['start'] + task['duration'] / 2, machine_id,
                        f'J{task["job"]}O{task["operation"]}',
                        ha='center', va='center', fontsize=8)

        ax.set_xlabel('时间')
        ax.set_ylabel('机器')
        ax.set_title('机器调度甘特图')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_agv_gantt(schedule: Dict, save_path: Optional[str] = None):
        """绘制AGV调度甘特图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.Set3(np.linspace(0, 1, 10))

        agv_tasks = {}
        for key, task_info in schedule['job_schedule'].items():
            agv_id = task_info['agv']
            if agv_id not in agv_tasks:
                agv_tasks[agv_id] = []

            agv_tasks[agv_id].append({
                'job': key[0],
                'operation': key[1],
                'start': task_info['transport'][0],
                'end': task_info['transport'][1],
                'duration': task_info['transport'][1] - task_info['transport'][0]
            })

        # 绘制每个AGV的任务
        for agv_id, tasks in agv_tasks.items():
            for task in tasks:
                ax.barh(agv_id, task['duration'],
                        left=task['start'], height=0.6,
                        color=colors[task['job'] % len(colors)],
                        edgecolor='black', alpha=0.7)

                # 添加文本标签
                ax.text(task['start'] + task['duration'] / 2, agv_id,
                        f'J{task["job"]}O{task["operation"]}',
                        ha='center', va='center', fontsize=8)

        ax.set_xlabel('时间')
        ax.set_ylabel('AGV')
        ax.set_title('AGV调度甘特图')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_pareto_front(solutions: List, objectives: List[str], save_path: Optional[str] = None):
        """绘制帕累托前沿"""
        if len(objectives) != 3:
            print("目前只支持3目标帕累托前沿可视化")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 提取目标值
        x = [sol[0] for sol in solutions]
        y = [sol[1] for sol in solutions]
        z = [sol[2] for sol in solutions]

        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.7)

        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[2])
        ax.set_title('帕累托前沿')

        plt.colorbar(scatter, ax=ax, label=objectives[2])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()