"""算法配置参数"""

# 编码方案配置
ENCODING_CONFIG = {
    'initialization_methods': ['random', 'neh', 'spt', 'lpt'],
    'default_method': 'random',
    'max_operations': 1000,  # 最大工序数限制
}

# 解码器配置
DECODER_CONFIG = {
    'battery_management': True,
    'charging_strategy': 'immediate',  # immediate, delayed
    'scheduling_policy': 'active',  # active, semi-active
}

# 目标函数权重
OBJECTIVE_WEIGHTS = {
    'makespan': 0.4,
    'energy': 0.3,
    'cost': 0.3
}

# 性能参数
PERFORMANCE_CONFIG = {
    'max_decode_time': 1.0,  # 最大解码时间（秒）
    'memory_limit_mb': 1024,  # 内存限制
}