from typing import Dict, Any, List
from .base_algorithm import BaseAlgorithm
from .metaheuristic.nsga2 import NSGA2
from .reinforcement_learning.q_learning import QLearning


class HybridAlgorithm(BaseAlgorithm):
    """元启发式 + Q-Learning混合算法"""

    def __init__(self, model, config: Dict):
        super().__init__(model, config)
        self.metaheuristic_config = config.get('metaheuristic', {})
        self.rl_config = config.get('reinforcement_learning', {})

        self.metaheuristic = NSGA2(self.metaheuristic_config)
        self.rl_agent = QLearning(self.rl_config)

        # 定义RL状态和动作
        self.rl_states = ['early_search', 'mid_search', 'late_search']
        self.rl_actions = ['intensify', 'diversify', 'balance']

    def initialize(self):
        """初始化算法"""
        self.metaheuristic.initialize_population()
        print("✅ 混合算法初始化完成")

    def run(self):
        """运行混合算法"""
        print("🚀 开始运行混合算法...")

        for generation in range(self.config.get('max_generations', 100)):
            # 元启发式搜索
            population = self.metaheuristic.population
            fitness = self.metaheuristic.evaluate_population(population)

            # Q-Learning自适应调整
            state = self._get_search_state(population, fitness, generation)
            action = self.rl_agent.select_action(state)
            self._apply_action(action, population)

            # 执行元启发式操作
            selected = self.metaheuristic.selection(population, fitness)
            offspring = self.metaheuristic.crossover(selected)
            offspring = self.metaheuristic.mutation(offspring)

            # 更新种群
            self.metaheuristic.population = self.metaheuristic._create_new_population(selected, offspring)

            # 计算奖励并更新Q值
            reward = self._calculate_reward(population, self.metaheuristic.population)
            next_state = self._get_search_state(self.metaheuristic.population,
                                                self.metaheuristic.evaluate_population(self.metaheuristic.population),
                                                generation + 1)
            self.rl_agent.update(state, action, reward, next_state)

            # 记录收敛数据
            best_fitness = min([sum(f.values()) for f in fitness])
            self.convergence_data.append(best_fitness)

            if generation % 10 == 0:
                print(f"📊 第{generation}代, 最佳适应度: {best_fitness:.4f}")

        # 获取最终解
        self.solution = self._extract_solution()
        print("✅ 混合算法运行完成")

    def _get_search_state(self, population: List, fitness: List, generation: int) -> str:
        """获取搜索状态"""
        # 基于种群多样性和搜索进度定义状态
        diversity = self._calculate_diversity(population)
        progress = generation / self.config.get('max_generations', 100)

        if progress < 0.33:
            return 'early_search'
        elif progress < 0.66:
            return 'mid_search'
        else:
            return 'late_search'

    def _calculate_diversity(self, population: List) -> float:
        """计算种群多样性"""
        if len(population) <= 1:
            return 0.0

        # 简化实现：计算解之间的平均距离
        total_distance = 0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # 这里需要根据具体解的结构计算距离
                total_distance += 1.0  # 简化
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _apply_action(self, action: str, population: List):
        """应用RL选择的动作"""
        if action == 'intensify':
            # 强化局部搜索
            self.metaheuristic.mutation_rate *= 0.8
        elif action == 'diversify':
            # 增加多样性
            self.metaheuristic.mutation_rate *= 1.2
        elif action == 'balance':
            # 保持平衡
            pass

        # 确保参数在合理范围内
        self.metaheuristic.mutation_rate = max(0.01, min(0.5, self.metaheuristic.mutation_rate))

    def _calculate_reward(self, old_population: List, new_population: List) -> float:
        """计算奖励"""
        old_fitness = [sum(f.values()) for f in self.metaheuristic.evaluate_population(old_population)]
        new_fitness = [sum(f.values()) for f in self.metaheuristic.evaluate_population(new_population)]

        old_best = min(old_fitness)
        new_best = min(new_fitness)

        # 奖励基于改进程度
        improvement = old_best - new_best  # 负值表示改进
        return improvement * 10  # 缩放奖励

    def _extract_solution(self) -> Dict:
        """从最终种群中提取最佳解"""
        fitness = self.metaheuristic.evaluate_population(self.metaheuristic.population)
        best_idx = self.metaheuristic._find_best_solution(fitness)
        return self.metaheuristic.population[best_idx]

    def get_solution(self) -> Dict:
        """获取解"""
        return self.solution