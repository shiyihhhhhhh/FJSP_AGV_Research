import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt

from .base_model import BaseModel


class FJSPAGVModel(BaseModel):
    """考虑AGV电池约束的柔性作业车间绿色集成调度模型"""

    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)
        self.model = gp.Model("FJSP_AGV")
        self.theoretical_bounds = {}

        # 变量存储
        self.variables = {}

    def calculate_theoretical_bounds(self) -> Dict[str, float]:
        """计算理论下界"""
        data = self.data
        n, m, v = data['n'], data['m'], data['v']
        o_j = data['o_j']

        # 1. 完工时间下界
        critical_path = 0
        for j in range(n):
            job_time = sum(min(data['p_jhi'].get((j, h, i), 1000) for i in range(m))
                           for h in range(o_j[j]))
            critical_path = max(critical_path, job_time)

        total_workload = sum(min(data['p_jhi'].get((j, h, i), 1000) for i in range(m))
                             for j in range(n) for h in range(o_j[j]))
        machine_load = total_workload / m if m > 0 else 0
        makespan_lb = max(critical_path, machine_load)

        # 2. 能耗下界
        energy_lb = sum(min(data['p_jhi'].get((j, h, i), 1000) *
                            data['e_machine_jhi'].get((j, h, i), 1.0)
                            for i in range(m))
                        for j in range(n) for h in range(o_j[j]))

        # 3. 成本下界
        cost_lb = 0
        for j in range(n):
            for h in range(o_j[j]):
                costs = [data['c_jhi'].get((j, h, i), 1.5) for i in range(m)]
                avg_cost = sum(costs) / len(costs)
                cost_lb += avg_cost * 0.8

        transport_cost_lb = sum(data['tt_jh'].get((j, h), 2) * 0.02
                                for j in range(n) for h in range(o_j[j]))

        cost_lb = cost_lb + transport_cost_lb
        cost_lb = max(cost_lb, 15)

        self.theoretical_bounds = {
            'makespan': makespan_lb,
            'energy': energy_lb,
            'cost': cost_lb
        }

        return self.theoretical_bounds

    def create_variables(self):
        """创建决策变量"""
        data = self.data
        n, m, v = data['n'], data['m'], data['v']
        o_j = data['o_j']

        print("📝 创建决策变量...")

        # 1. 基本变量
        self.variables['C_max'] = self.model.addVar(vtype=GRB.CONTINUOUS, name="C_max")

        # 2. 加工调度变量
        self.variables['s'] = {}
        self.variables['c'] = {}
        self.variables['x'] = {}

        for j in range(n):
            for h in range(o_j[j]):
                self.variables['s'][j, h] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, name=f"s_{j}_{h}"
                )
                self.variables['c'][j, h] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, name=f"c_{j}_{h}"
                )

                for i in range(m):
                    self.variables['x'][j, h, i] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"x_{j}_{h}_{i}"
                    )

        # 3. AGV调度变量
        self.variables['ts'] = {}
        self.variables['tc'] = {}
        self.variables['y'] = {}
        self.variables['task_battery'] = {}

        for j in range(n):
            for h in range(o_j[j]):
                self.variables['ts'][j, h] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, name=f"ts_{j}_{h}"
                )
                self.variables['tc'][j, h] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, name=f"tc_{j}_{h}"
                )

                for v_idx in range(v):
                    self.variables['y'][j, h, v_idx] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"y_{j}_{h}_{v_idx}"
                    )
                    self.variables['task_battery'][j, h, v_idx] = self.model.addVar(
                        lb=data['B_min'][v_idx], ub=data['B_max'][v_idx],
                        vtype=GRB.CONTINUOUS, name=f"b_{j}_{h}_{v_idx}"
                    )

        # 4. 目标函数变量
        self.variables['total_energy'] = self.model.addVar(
            vtype=GRB.CONTINUOUS, name="total_energy"
        )
        self.variables['total_cost'] = self.model.addVar(
            vtype=GRB.CONTINUOUS, name="total_cost"
        )

        # 5. 负载均衡变量
        self.variables['machine_working_time'] = {}
        for i in range(m):
            self.variables['machine_working_time'][i] = self.model.addVar(
                vtype=GRB.CONTINUOUS, name=f"work_time_{i}"
            )

        self.variables['max_work_time'] = self.model.addVar(
            vtype=GRB.CONTINUOUS, name="max_work_time"
        )
        self.variables['min_work_time'] = self.model.addVar(
            vtype=GRB.CONTINUOUS, name="min_work_time"
        )
        self.variables['load_balance'] = self.model.addVar(
            vtype=GRB.CONTINUOUS, name="load_balance"
        )

        print("✅ 决策变量创建完成")

    def set_objective(self, weights: Optional[List[float]] = None):
        """设置动态标度化的多目标函数"""
        if weights is None:
            weights = [0.4, 0.3, 0.3]

        bounds = self.calculate_theoretical_bounds()

        # 上界估计
        makespan_ub = bounds['makespan'] * 2.0
        energy_ub = bounds['energy'] * 1.5
        cost_ub = bounds['cost'] * 2.5

        # 标度化目标函数
        scaled_makespan = self.variables['C_max'] / makespan_ub
        scaled_energy = self.variables['total_energy'] / energy_ub
        scaled_cost = self.variables['total_cost'] / cost_ub

        # 加权目标
        objective = (
                weights[0] * scaled_makespan +
                weights[1] * scaled_energy +
                weights[2] * scaled_cost
        )

        self.model.setObjective(objective, GRB.MINIMIZE)
        print("✅ 目标函数设置完成")

    def add_constraints(self):
        """添加完整的约束体系"""
        print("🔧 添加约束条件...")

        self._add_basic_assignment_constraints()
        self._add_time_constraints()
        self._add_resource_conflict_constraints()
        self._add_battery_constraints()
        self._add_load_balance_constraints()
        self._add_objective_constraints()

        print("✅ 所有约束添加完成")

    def _add_basic_assignment_constraints(self):
        """添加基本分配约束"""
        data = self.data
        n, m, v = data['n'], data['m'], data['v']
        o_j = data['o_j']

        for j in range(n):
            for h in range(o_j[j]):
                # 机器分配约束
                self.model.addConstr(
                    gp.quicksum(self.variables['x'][j, h, i] for i in range(m)) == 1,
                    f"assign_{j}_{h}"
                )
                # AGV分配约束
                self.model.addConstr(
                    gp.quicksum(self.variables['y'][j, h, v_idx] for v_idx in range(v)) == 1,
                    f"agv_assign_{j}_{h}"
                )

    def _add_time_constraints(self):
        """添加时间相关约束"""
        data = self.data
        n, m, v = data['n'], data['m'], data['v']
        o_j, big_M = data['o_j'], data['big_M']

        # 加工时间约束
        for j in range(n):
            for h in range(o_j[j]):
                for i in range(m):
                    processing_time = data['p_jhi'].get((j, h, i), 3)
                    self.model.addConstr(
                        self.variables['c'][j, h] >= self.variables['s'][j, h] + processing_time -
                        big_M * (1 - self.variables['x'][j, h, i]),
                        f"process_time_{j}_{h}_{i}"
                    )

        # 运输时间约束
        for j in range(n):
            for h in range(o_j[j]):
                for v_idx in range(v):
                    transport_time = data['tt_jh'].get((j, h), 2)
                    self.model.addConstr(
                        self.variables['tc'][j, h] >= self.variables['ts'][j, h] + transport_time -
                        big_M * (1 - self.variables['y'][j, h, v_idx]),
                        f"transport_time_{j}_{h}_{v_idx}"
                    )

        # 协调约束
        for j in range(n):
            for h in range(o_j[j]):
                self.model.addConstr(
                    self.variables['s'][j, h] >= self.variables['tc'][j, h],
                    f"coordination_{j}_{h}"
                )

        # 工序顺序约束
        for j in range(n):
            for h in range(1, o_j[j]):
                self.model.addConstr(
                    self.variables['s'][j, h] >= self.variables['c'][j, h - 1],
                    f"precedence_{j}_{h}"
                )

        self._add_agv_sequence_constraints()

    def _add_agv_sequence_constraints(self):
        """添加AGV任务顺序约束"""
        data = self.data
        n, m, v = data['n'], data['m'], data['v']
        o_j, big_M = data['o_j'], data['big_M']

        for v_idx in range(v):
            tasks = [(j, h) for j in range(n) for h in range(o_j[j])]
            tasks_sorted = sorted(tasks, key=lambda x: (x[0], x[1]))

            for i in range(len(tasks_sorted) - 1):
                j1, h1 = tasks_sorted[i]
                j2, h2 = tasks_sorted[i + 1]

                self.model.addConstr(
                    self.variables['ts'][j2, h2] >= self.variables['tc'][j1, h1] -
                    big_M * (2 - self.variables['y'][j1, h1, v_idx] - self.variables['y'][j2, h2, v_idx]),
                    f"agv_sequence_{v_idx}_{j1}_{h1}_{j2}_{h2}"
                )

    def _add_resource_conflict_constraints(self):
        """添加资源冲突约束"""
        data = self.data
        n, m, v = data['n'], data['m'], data['v']
        o_j, big_M = data['o_j'], data['big_M']

        # 机器冲突约束
        for i in range(m):
            tasks = [(j, h) for j in range(n) for h in range(o_j[j])]
            for idx1 in range(len(tasks)):
                j1, h1 = tasks[idx1]
                for idx2 in range(idx1 + 1, len(tasks)):
                    j2, h2 = tasks[idx2]
                    if j1 == j2:
                        continue

                    V = self.model.addVar(vtype=GRB.BINARY, name=f"V_{j1}_{h1}_{j2}_{h2}_{i}")

                    self.model.addConstr(
                        self.variables['s'][j2, h2] >= self.variables['c'][j1, h1] -
                        big_M * (3 - self.variables['x'][j1, h1, i] - self.variables['x'][j2, h2, i] - V),
                        f"machine_conflict1_{j1}_{h1}_{j2}_{h2}_{i}"
                    )
                    self.model.addConstr(
                        self.variables['s'][j1, h1] >= self.variables['c'][j2, h2] -
                        big_M * (2 - self.variables['x'][j1, h1, i] - self.variables['x'][j2, h2, i] + V),
                        f"machine_conflict2_{j1}_{h1}_{j2}_{h2}_{i}"
                    )

    def _add_battery_constraints(self):
        """添加电量约束"""
        data = self.data
        n, m, v = data['n'], data['m'], data['v']
        o_j, big_M = data['o_j'], data['big_M']

        # 初始电量设置
        initial_battery = data['B_max'][0] * 0.8

        for v_idx in range(v):
            first_task_found = False
            for j in range(n):
                for h in range(o_j[j]):
                    if not first_task_found:
                        self.model.addConstr(
                            self.variables['task_battery'][j, h, v_idx] == initial_battery,
                            f"initial_battery_{v_idx}"
                        )
                        first_task_found = True
                    else:
                        self.model.addConstr(
                            self.variables['task_battery'][j, h, v_idx] >= data['B_min'][v_idx],
                            f"min_battery_{j}_{h}_{v_idx}"
                        )

        # 电量递减约束
        for v_idx in range(v):
            tasks = [(j, h) for j in range(n) for h in range(o_j[j])]
            tasks_sorted = sorted(tasks, key=lambda x: (x[0], x[1]))

            for i in range(len(tasks_sorted) - 1):
                j1, h1 = tasks_sorted[i]
                j2, h2 = tasks_sorted[i + 1]

                energy_consumed = data['e_AGV_jh'][j1, h1] * data['tt_jh'][j1, h1]
                self.model.addConstr(
                    self.variables['task_battery'][j2, h2, v_idx] <=
                    self.variables['task_battery'][j1, h1, v_idx] - energy_consumed +
                    big_M * (2 - self.variables['y'][j1, h1, v_idx] - self.variables['y'][j2, h2, v_idx]),
                    f"battery_decrease_{v_idx}_{j1}_{h1}_{j2}_{h2}"
                )

    def _add_load_balance_constraints(self):
        """添加负载均衡约束"""
        data = self.data
        n, m = data['n'], data['m']
        o_j = data['o_j']

        for i in range(m):
            self.model.addConstr(
                self.variables['machine_working_time'][i] == gp.quicksum(
                    self.variables['x'][j, h, i] * data['p_jhi'][j, h, i]
                    for j in range(n) for h in range(o_j[j])
                ),
                f"work_time_{i}"
            )

        for i in range(m):
            self.model.addConstr(self.variables['max_work_time'] >= self.variables['machine_working_time'][i])
            self.model.addConstr(self.variables['min_work_time'] <= self.variables['machine_working_time'][i])

        self.model.addConstr(
            self.variables['load_balance'] == self.variables['max_work_time'] - self.variables['min_work_time'],
            "load_balance_def"
        )

    def _add_objective_constraints(self):
        """添加目标函数计算约束"""
        data = self.data
        n, m, v = data['n'], data['m'], data['v']
        o_j = data['o_j']

        # 最大完工时间
        for j in range(n):
            last_op = o_j[j] - 1
            self.model.addConstr(self.variables['C_max'] >= self.variables['c'][j, last_op], f"C_max_{j}")

        # 总能耗计算
        machine_energy = gp.quicksum(
            self.variables['x'][j, h, i] * data['p_jhi'].get((j, h, i), 3) * data['e_machine_jhi'].get((j, h, i), 1.0)
            for j in range(n) for h in range(o_j[j]) for i in range(m)
        )
        agv_energy = gp.quicksum(
            self.variables['y'][j, h, v_idx] * data['tt_jh'].get((j, h), 2) * data['e_AGV_jh'].get((j, h), 0.1)
            for j in range(n) for h in range(o_j[j]) for v_idx in range(v)
        )
        self.model.addConstr(self.variables['total_energy'] == machine_energy + agv_energy, "total_energy_constr")

        # 总成本计算
        machine_cost = gp.quicksum(
            self.variables['x'][j, h, i] * data['p_jhi'].get((j, h, i), 3) * data['c_jhi'].get((j, h, i), 1.2)
            for j in range(n) for h in range(o_j[j]) for i in range(m)
        )
        agv_cost = gp.quicksum(
            self.variables['y'][j, h, v_idx] * data['tt_jh'].get((j, h), 2) * 0.01
            for j in range(n) for h in range(o_j[j]) for v_idx in range(v)
        )
        self.model.addConstr(self.variables['total_cost'] == machine_cost + agv_cost, "total_cost_constr")

    def solve(self, time_limit: int = 300, output_flag: bool = True) -> bool:
        """求解模型"""
        print(f"🚀 开始求解模型...")

        self.model.setParam('OutputFlag', 1 if output_flag else 0)
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', 0.05)
        self.model.setParam('Presolve', 1)
        self.model.setParam('Cuts', 1)

        start_time = time.time()
        self.model.optimize()
        self.solve_time = time.time() - start_time

        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            if self.model.SolCount > 0:
                print("✅ 找到可行解！")
                self._extract_solution()
                self._calculate_performance_metrics()
                return True

        print(f"❌ 求解失败，状态: {self.model.status}")
        return False

    def _extract_solution(self):
        """提取解信息"""
        data = self.data
        n, o_j = data['n'], data['o_j']

        self.solution = {
            'makespan': self.variables['C_max'].X,
            'energy': self.variables['total_energy'].X,
            'cost': self.variables['total_cost'].X,
            'load_balance': self.variables['load_balance'].X if hasattr(self.variables['load_balance'], 'X') else 0,
            'machine_assignments': {},
            'agv_assignments': {},
            'schedule': {},
            'times': [],
            'battery_levels': {}
        }

        # 提取机器分配
        for j in range(n):
            for h in range(o_j[j]):
                for i in range(data['m']):
                    if self.variables['x'][j, h, i].X > 0.5:
                        self.solution['machine_assignments'][(j, h)] = i
                        self.solution['schedule'][(j, h, 'machine')] = {
                            'machine': i,
                            'start': self.variables['s'][j, h].X,
                            'end': self.variables['c'][j, h].X
                        }
                        self.solution['times'].append(self.variables['s'][j, h].X)
                        self.solution['times'].append(self.variables['c'][j, h].X)
                        break

        # 提取AGV分配
        for j in range(n):
            for h in range(o_j[j]):
                for v_idx in range(data['v']):
                    if self.variables['y'][j, h, v_idx].X > 0.5:
                        self.solution['agv_assignments'][(j, h)] = v_idx
                        battery_level = self.variables['task_battery'][j, h, v_idx].X
                        self.solution['schedule'][(j, h, 'transport')] = {
                            'agv': v_idx,
                            'start': self.variables['ts'][j, h].X,
                            'end': self.variables['tc'][j, h].X,
                            'battery': battery_level
                        }
                        self.solution['battery_levels'][(j, h)] = battery_level
                        self.solution['times'].append(self.variables['ts'][j, h].X)
                        self.solution['times'].append(self.variables['tc'][j, h].X)
                        break

    def _calculate_performance_metrics(self):
        """计算性能指标"""
        bounds = self.theoretical_bounds
        solution = self.solution

        makespan_gap = (solution['makespan'] - bounds['makespan']) / max(bounds['makespan'], 1e-6)
        energy_gap = (solution['energy'] - bounds['energy']) / max(bounds['energy'], 1e-6)
        cost_gap = (solution['cost'] - bounds['cost']) / max(bounds['cost'], 1e-6)

        self.performance_metrics = {
            'makespan_gap': makespan_gap,
            'energy_gap': energy_gap,
            'cost_gap': cost_gap,
            'solve_time': self.solve_time,
            'objective_value': self.model.ObjVal,
            'mip_gap': self.model.MIPGap,
            'node_count': self.model.NodeCount
        }

    def plot_schedule(self, save_path: Optional[str] = None):
        """绘制调度甘特图"""
        if not self.solution:
            print("❌ 无解数据，无法绘图")
            return

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体或DejaVu Sans
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 机器调度图
        colors = plt.cm.Set3(np.linspace(0, 1, self.data['n']))
        machine_tasks = {key: info for key, info in self.solution['schedule'].items() if key[2] == 'machine'}

        for (j, h, _), info in machine_tasks.items():
            ax1.barh(info['machine'], info['end'] - info['start'],
                     left=info['start'], height=0.6,
                     color=colors[j], label=f'Job{j}Op{h}')

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Machine')
        ax1.set_title('Machine Scheduling Gantt Chart')
        ax1.grid(True, alpha=0.3)

        # AGV调度图
        agv_tasks = {key: info for key, info in self.solution['schedule'].items() if key[2] == 'transport'}

        for (j, h, _), info in agv_tasks.items():
            ax2.barh(info['agv'], info['end'] - info['start'],
                     left=info['start'], height=0.6,
                     color=colors[j], label=f'Job{j}Op{h}')

        ax2.set_xlabel('Time')
        ax2.set_ylabel('AGV')
        ax2.set_title('AGV Scheduling Gantt Chart')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            # 创建目录如果不存在
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 调度图已保存至: {save_path}")

        plt.show()