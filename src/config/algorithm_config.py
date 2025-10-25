from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class AlgorithmConfig:
    """算法配置类"""
    population_size: int = 100
    max_generations: int = 500
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    weights: List[float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = [0.4, 0.3, 0.3]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'population_size': self.population_size,
            'max_generations': self.max_generations,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'weights': self.weights
        }