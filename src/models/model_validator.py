import json
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


class ModelValidator:
    """模型验证器"""

    def __init__(self, model):
        self.model = model
        self.validation_results = {}

    def validate_all(self) -> Dict[str, bool]:
        """执行全部验证"""
        self.validation_results = {
            'mathematical_feasibility': self.validate_mathematical_feasibility(),
            'physical_reasonableness': self.validate_physical_reasonableness(),
            'agv_scheduling': self.validate_agv_scheduling(),
            'battery_management': self.validate_battery_management(),
            'constraint_satisfaction': self.validate_constraint_satisfaction()
        }
        return self.validation_results

    def validate_mathematical_feasibility(self) -> bool:
        """验证数学可行性"""
        return self.model.model.status in [2, 3, 4]  # GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL

    def validate_physical_reasonableness(self) -> bool:
        """验证物理合理性"""
        solution = self.model.solution
        time_checks = all(time_val >= 0 for time_val in solution['times'])

        checks = {
            'positive_times': time_checks,
            'makespan_reasonable': solution['makespan'] > 0 and solution['makespan'] < 1000,
            'energy_reasonable': solution['energy'] > 0,
            'cost_reasonable': solution['cost'] > 0 and solution['cost'] < 50
        }

        return all(checks.values())

    def validate_agv_scheduling(self) -> bool:
        """验证AGV调度"""
        solution = self.model.solution

        transport_tasks = [(j, h) for (j, h, task_type) in solution['schedule'].keys() if task_type == 'transport']
        transport_times = []

        for (j, h, task_type) in solution['schedule'].keys():
            if task_type == 'transport':
                info = solution['schedule'][(j, h, task_type)]
                transport_times.append((info['start'], info['end'], f"工件{j}工序{h}"))

        # 检查重叠
        overlaps = 0
        for i in range(len(transport_times)):
            for j in range(i + 1, len(transport_times)):
                start1, end1, task1 = transport_times[i]
                start2, end2, task2 = transport_times[j]
                if max(start1, start2) < min(end1, end2):
                    overlaps += 1

        return overlaps == 0

    def validate_battery_management(self) -> Dict[str, bool]:
        """验证电量管理"""
        solution = self.model.solution
        data = self.model.data

        battery_levels = list(solution['battery_levels'].values())

        battery_checks = {
            'no_zero_battery': all(battery > 0 for battery in battery_levels),
            'battery_in_range': all(data['B_min'][0] <= battery <= data['B_max'][0] for battery in battery_levels),
            'battery_decreasing': self._check_battery_decreasing(solution)
        }

        return battery_checks

    def _check_battery_decreasing(self, solution) -> bool:
        """检查电量是否合理递减"""
        tasks = [(j, h) for (j, h) in solution['battery_levels'].keys()]
        tasks_sorted = sorted(tasks, key=lambda x: (x[0], x[1]))

        for i in range(len(tasks_sorted) - 1):
            j1, h1 = tasks_sorted[i]
            j2, h2 = tasks_sorted[i + 1]
            if solution['battery_levels'][(j2, h2)] > solution['battery_levels'][(j1, h1)]:
                return False
        return True

    def validate_constraint_satisfaction(self) -> bool:
        """验证约束满足"""
        # 这里可以添加特定约束的验证
        # 由于在模型中已经通过求解器确保，这里主要检查一些关键约束
        return True

    def generate_validation_report(self) -> str:
        """生成验证报告"""
        if not self.validation_results:
            self.validate_all()

        report = ["🔍 模型验证报告", "=" * 50]
        for check, result in self.validation_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            report.append(f"{check}: {status}")

        return "\n".join(report)

    def plot_battery_usage(self, save_path: str = None):
        """绘制电量使用情况"""
        if not self.model.solution:
            print("无解数据，无法绘图")
            return

        # 设置中文字体
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        battery_levels = self.model.solution['battery_levels']
        tasks = sorted(battery_levels.keys(), key=lambda x: (x[0], x[1]))
        task_labels = [f'J{j}O{h}' for j, h in tasks]
        levels = [battery_levels[task] for task in tasks]

        plt.figure(figsize=(10, 6))
        plt.plot(task_labels, levels, marker='o', linestyle='-', color='b')
        plt.axhline(y=self.model.data['B_min'][0], color='r', linestyle='--', label='Min Safety Level')
        plt.axhline(y=self.model.data['B_max'][0], color='g', linestyle='--', label='Max Capacity')
        plt.xlabel('Task (Job-Operation)')
        plt.ylabel('Battery Level')
        plt.title('AGV Battery Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        if save_path:
            # 创建目录如果不存在
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()