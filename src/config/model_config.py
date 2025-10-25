from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ModelConfig:
    """模型配置类"""
    n_jobs: int = 3
    n_machines: int = 2
    n_agvs: int = 1
    operations_per_job: List[int] = field(default_factory=lambda: [2, 2, 2])
    battery_capacity: int = 50
    safety_battery: int = 10
    big_M: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'n': self.n_jobs,
            'm': self.n_machines,
            'v': self.n_agvs,
            'o_j': self.operations_per_job,
            'B_max': [self.battery_capacity] * self.n_agvs,
            'B_min': [self.safety_battery] * self.n_agvs,
            'big_M': self.big_M
        }