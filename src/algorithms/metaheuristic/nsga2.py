import numpy as np
from typing import List, Dict, Tuple, Any
import random
# 使用相对导入避免循环
from .base_metaheuristic import BaseMetaheuristic, Chromosome


class ScheduleDecoder:
    """调度解码器 - 活动调度生成器"""

    def __init__(self, problem_data: Dict[str, Any]):
        self.problem_data = problem_data

    def decode(self, chromosome: Chromosome) -> Dict[str, Any]:
        """将染色体解码为可行的调度方案"""
        # 这里保持原有的ScheduleDecoder实现
        # 初始化时间线
        machine_timelines = {i: [] for i in range(self.problem_data['n_machines'])}
        agv_timelines = {i: [] for i in range(self.problem_data['n_agvs'])}
        agv_batteries = {i: self.problem_data['battery_capacity'] for i in range(self.problem_data['n_agvs'])}

        # 记录每个工件的当前状态
        job_states = {
            job_id: {
                'current_time': self.problem_data['release_time'].get(job_id, 0),
                'completed_operations': 0,
                'current_location': None
            }
            for job_id in range(self.problem_data['n_jobs'])
        }

        schedule = {}
        operation_counter = {job_id: 0 for job_id in range(self.problem_data['n_jobs'])}

        # 按OS顺序安排工序
        for job_id in chromosome.OS:
            op_id = operation_counter[job_id]
            operation_counter[job_id] += 1

            # 获取分配的机器和AGV
            if op_id < len(chromosome.MA[job_id]):
                machine_id, agv_id = chromosome.MA[job_id][op_id]
            else:
                machine_id = 0
                agv_id = 0

            # 1. 计算运输时间窗口
            transport_info = self._schedule_transport(
                job_id, op_id, agv_id, job_states[job_id],
                agv_timelines[agv_id], agv_batteries[agv_id]
            )

            # 2. 计算加工时间窗口
            process_info = self._schedule_processing(
                job_id, op_id, machine_id, transport_info['end_time'],
                machine_timelines[machine_id]
            )

            # 3. 更新状态
            self._update_state(
                job_id, op_id, machine_id, agv_id, transport_info, process_info,
                job_states, machine_timelines, agv_timelines, agv_batteries, schedule
            )

        # 计算目标函数值
        objectives = self._calculate_objectives(schedule)

        chromosome.schedule = {
            'machine_timelines': machine_timelines,
            'agv_timelines': agv_timelines,
            'job_schedule': schedule,
            'objectives': objectives
        }
        chromosome.objectives = objectives

        return chromosome.schedule

    # 这里包含所有原有的ScheduleDecoder方法...
    def _schedule_transport(self, job_id: int, op_id: int, agv_id: int,
                            job_state: Dict, agv_timeline: List, agv_battery: float) -> Dict:
        # 保持原有实现
        transport_time = self.problem_data['transport_time'].get((job_id, op_id), 1.0)
        transport_energy = self.problem_data['transport_energy'].get((job_id, op_id), 0.1) * transport_time

        earliest_start = job_state['current_time']

        if agv_timeline:
            last_task_end = max([task[1] for task in agv_timeline]) if agv_timeline else 0
            earliest_start = max(earliest_start, last_task_end)

        if agv_battery - transport_energy < self.problem_data['min_safety_battery']:
            charge_time = (self.problem_data['battery_capacity'] - agv_battery) / self.problem_data['charging_rate']
            earliest_start += charge_time
            transport_start = earliest_start
            battery_after_charge = self.problem_data['battery_capacity']
        else:
            transport_start = earliest_start
            battery_after_charge = agv_battery

        transport_end = transport_start + transport_time
        battery_after_transport = battery_after_charge - transport_energy

        return {
            'start_time': transport_start,
            'end_time': transport_end,
            'energy_consumed': transport_energy,
            'battery_after': battery_after_transport
        }

    def _schedule_processing(self, job_id: int, op_id: int, machine_id: int,
                             transport_end_time: float, machine_timeline: List) -> Dict:
        processing_time = self.problem_data['processing_time'].get((job_id, op_id, machine_id), 2.0)
        processing_energy = self.problem_data['processing_energy'].get((job_id, op_id, machine_id),
                                                                       1.0) * processing_time

        earliest_start = transport_end_time

        if machine_timeline:
            last_task_end = max([task[1] for task in machine_timeline]) if machine_timeline else 0
            earliest_start = max(earliest_start, last_task_end)

        process_start = earliest_start
        process_end = process_start + processing_time

        return {
            'start_time': process_start,
            'end_time': process_end,
            'energy_consumed': processing_energy
        }

    def _update_state(self, job_id: int, op_id: int, machine_id: int, agv_id: int,
                      transport_info: Dict, process_info: Dict, job_states: Dict,
                      machine_timelines: Dict, agv_timelines: Dict, agv_batteries: Dict,
                      schedule: Dict):
        job_states[job_id]['current_time'] = process_info['end_time']
        job_states[job_id]['completed_operations'] += 1
        job_states[job_id]['current_location'] = machine_id

        machine_timelines[machine_id].append((
            process_info['start_time'],
            process_info['end_time'],
            f"P_{job_id}_{op_id}"
        ))

        agv_timelines[agv_id].append((
            transport_info['start_time'],
            transport_info['end_time'],
            f"T_{job_id}_{op_id}"
        ))

        agv_batteries[agv_id] = transport_info['battery_after']

        schedule[(job_id, op_id)] = {
            'machine': machine_id,
            'agv': agv_id,
            'transport': (transport_info['start_time'], transport_info['end_time']),
            'process': (process_info['start_time'], process_info['end_time']),
            'transport_energy': transport_info['energy_consumed'],
            'process_energy': process_info['energy_consumed']
        }

    def _calculate_objectives(self, schedule: Dict) -> Tuple[float, float, float]:
        makespan = 0
        total_energy = 0
        total_cost = 0

        for op_key, op_info in schedule.items():
            job_id, op_id = op_key
            machine_id = op_info['machine']

            makespan = max(makespan, op_info['process'][1])
            total_energy += op_info['transport_energy'] + op_info['process_energy']

            processing_cost_rate = self.problem_data['processing_cost'].get(
                (job_id, op_id, machine_id), 1.0
            )
            processing_cost = processing_cost_rate * (op_info['process'][1] - op_info['process'][0])
            transport_cost = 0.01 * (op_info['transport'][1] - op_info['transport'][0])
            total_cost += processing_cost + transport_cost

        return makespan, total_energy, total_cost