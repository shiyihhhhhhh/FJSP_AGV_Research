import numpy as np
from typing import Dict, List, Tuple, Any
import random


class FJSPAGVProblemAdapter:
    """将FJSPAGVModel的数据格式转换为算法需要的格式"""

    @staticmethod
    def from_model_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """从模型数据转换为算法数据格式"""

        n_jobs = data['n']
        n_machines = data['m']
        n_agvs = data['v']
        n_operations = data['o_j']

        # 构建加工参数
        processing_time = {}
        processing_energy = {}
        processing_cost = {}

        for j in range(n_jobs):
            for h in range(n_operations[j]):
                for i in range(n_machines):
                    key = (j, h, i)
                    if key in data['p_jhi']:
                        processing_time[key] = data['p_jhi'][key]
                        processing_energy[key] = data['e_machine_jhi'].get(key, 1.0)
                        processing_cost[key] = data['c_jhi'].get(key, 1.0)

        # 构建运输参数
        transport_time = {}
        transport_energy = {}
        job_weight = {}

        for j in range(n_jobs):
            for h in range(n_operations[j]):
                key = (j, h)
                transport_time[key] = data['tt_jh'].get(key, 1.0)
                transport_energy[key] = data['e_AGV_jh'].get(key, 0.1)
                job_weight[key] = 10.0  # 默认重量

        # AGV参数
        battery_capacity = data['B_max'][0] if data['B_max'] else 100.0
        min_safety_battery = data['B_min'][0] if data['B_min'] else 20.0
        charging_rate = 10.0
        agv_capacity = 50.0

        # 时间参数
        release_time = data.get('release_time', {j: 0 for j in range(n_jobs)})
        due_date = data.get('due_date', {j: 100 for j in range(n_jobs)})

        return {
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'n_agvs': n_agvs,
            'n_operations': n_operations,
            'processing_time': processing_time,
            'processing_energy': processing_energy,
            'processing_cost': processing_cost,
            'transport_time': transport_time,
            'transport_energy': transport_energy,
            'job_weight': job_weight,
            'battery_capacity': battery_capacity,
            'min_safety_battery': min_safety_battery,
            'charging_rate': charging_rate,
            'agv_capacity': agv_capacity,
            'release_time': release_time,
            'due_date': due_date
        }

    @staticmethod
    def create_test_instance() -> Dict[str, Any]:
        """创建测试实例，与您模型的数据格式一致"""
        return {
            'n': 3,
            'm': 2,
            'v': 1,
            'o_j': [2, 2, 1],
            'p_jhi': {
                (0, 0, 0): 4, (0, 0, 1): 3,
                (0, 1, 0): 2, (0, 1, 1): 3,
                (1, 0, 0): 3, (1, 0, 1): 2,
                (1, 1, 0): 4, (1, 1, 1): 3,
                (2, 0, 0): 2, (2, 0, 1): 3
            },
            'e_machine_jhi': {
                (0, 0, 0): 1.0, (0, 0, 1): 1.2,
                (0, 1, 0): 0.8, (0, 1, 1): 1.0,
                (1, 0, 0): 1.1, (1, 0, 1): 0.9,
                (1, 1, 0): 1.0, (1, 1, 1): 1.1,
                (2, 0, 0): 0.9, (2, 0, 1): 1.0
            },
            'c_jhi': {
                (0, 0, 0): 5, (0, 0, 1): 6,
                (0, 1, 0): 4, (0, 1, 1): 5,
                (1, 0, 0): 5, (1, 0, 1): 4,
                (1, 1, 0): 6, (1, 1, 1): 5,
                (2, 0, 0): 4, (2, 0, 1): 5
            },
            'tt_jh': {
                (0, 0): 1, (0, 1): 2,
                (1, 0): 1, (1, 1): 1,
                (2, 0): 1
            },
            'e_AGV_jh': {
                (0, 0): 0.2, (0, 1): 0.3,
                (1, 0): 0.2, (1, 1): 0.2,
                (2, 0): 0.1
            },
            'B_max': [100],
            'B_min': [20],
            'release_time': {0: 0, 1: 0, 2: 0},
            'due_date': {0: 20, 1: 25, 2: 15},
            'big_M': 1000
        }