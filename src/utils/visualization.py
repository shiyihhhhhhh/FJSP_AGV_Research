import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional


class Visualization:
    """可视化工具类"""

    @staticmethod
    def plot_gantt_chart(schedule: Dict, save_path: Optional[str] = None):
        """绘制调度甘特图"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 机器调度图
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        machine_tasks = {key: info for key, info in schedule.items() if 'machine' in str(key)}
        agv_tasks = {key: info for key, info in schedule.items() if 'transport' in str(key)}

        for i, (key, info) in enumerate(machine_tasks.items()):
            ax1.barh(info['machine'], info['end'] - info['start'],
                     left=info['start'], height=0.6,
                     color=colors[i % 10], label=f'Task{i}')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Machine')
        ax1.set_title('Machine Scheduling Gantt Chart')
        ax1.grid(True, alpha=0.3)

        # AGV调度图
        for i, (key, info) in enumerate(agv_tasks.items()):
            ax2.barh(info['agv'], info['end'] - info['start'],
                     left=info['start'], height=0.6,
                     color=colors[i % 10], label=f'Task{i}')

        ax2.set_xlabel('Time')
        ax2.set_ylabel('AGV')
        ax2.set_title('AGV Scheduling Gantt Chart')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_convergence(convergence_data: Dict[str, list], save_path: Optional[str] = None):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        for key, values in convergence_data.items():
            plt.plot(values, label=key)
        plt.xlabel('Iterations')
        plt.ylabel('Objective Value')
        plt.title('Algorithm Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()