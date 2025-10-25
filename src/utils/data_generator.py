import numpy as np
from typing import Dict, Any, List
from src.config.model_config import ModelConfig


class DataGenerator:
    """数据生成器"""

    @staticmethod
    def generate_from_config(config: ModelConfig) -> Dict[str, Any]:
        """根据配置生成测试数据"""
        np.random.seed(42)

        data = config.to_dict()

        # 生成加工参数
        p_jhi = {}
        e_machine_jhi = {}
        c_jhi = {}
        for j in range(config.n_jobs):
            for h in range(config.operations_per_job[j]):
                for i in range(config.n_machines):
                    p_jhi[j, h, i] = np.random.randint(2, 6)
                    e_machine_jhi[j, h, i] = np.random.uniform(0.5, 1.5)
                    c_jhi[j, h, i] = np.random.uniform(1.0, 2.5)

        # 生成AGV参数
        e_AGV_jh = {}
        tt_jh = {}
        q_jh = {}
        for j in range(config.n_jobs):
            for h in range(config.operations_per_job[j]):
                e_AGV_jh[j, h] = np.random.uniform(0.05, 0.1)
                tt_jh[j, h] = np.random.randint(1, 3)
                q_jh[j, h] = np.random.randint(1, 2)

        data.update({
            'p_jhi': p_jhi,
            'e_machine_jhi': e_machine_jhi,
            'c_jhi': c_jhi,
            'e_AGV_jh': e_AGV_jh,
            'tt_jh': tt_jh,
            'q_jh': q_jh,
            'r_j': [0] * config.n_jobs,
            'd_j': [np.random.randint(15, 25) for _ in range(config.n_jobs)],
            'pi_j': [1.0] * config.n_jobs,
            'Q_v': [5] * config.n_agvs
        })

        return data

    @staticmethod
    def generate_specific_instance(n_jobs: int, n_machines: int, n_agvs: int, operations_per_job: List[int]):
        """生成特定实例"""
        config = ModelConfig(
            n_jobs=n_jobs,
            n_machines=n_machines,
            n_agvs=n_agvs,
            operations_per_job=operations_per_job
        )
        return DataGenerator.generate_from_config(config)