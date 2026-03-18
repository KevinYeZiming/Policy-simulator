"""
城市物质行动者网络仿真框架 (Urban Material Actant Network Simulation Framework)
版本：2.0
核心特性：
- 城市为封装点位，通过物质能力向量交互
- 支持四类科技政策模拟（投资、规制、信息、基建）
- 可接入外部真实数据（算力、能源、经济指标）
- 内置执行差距、时滞效应、动态自适应政策
- 输出关键治理观测指标
"""

import abc
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# ===================== 基础数据类型 =====================

@dataclass
class CapabilityVector:
    """物质能力向量：描述城市的基础设施容量和资源禀赋（相对稳定）"""
    # 能源相关
    solar_capacity: float = 0.0          # 光伏装机容量 (MW)
    wind_capacity: float = 0.0            # 风电装机容量 (MW)
    grid_capacity: float = 0.0            # 电网总容量 (MW)
    battery_capacity: float = 0.0         # 储能容量 (MWh)
    
    # 数字基础设施
    computing_power: float = 0.0           # 总算力 (PFLOPS)
    computing_efficiency: float = 0.5      # 算力功耗比 (MW/PFLOPS)
    network_bandwidth: float = 0.0         # 网络带宽 (Gbps)
    
    # 资源环境
    land_availability: float = 0.0          # 可用土地比例 (0-1)
    water_stress: float = 0.0               # 水资源压力指数 (0-1)
    emission_factor: float = 0.5            # 电网碳排放因子 (kg CO2/kWh)
    
    # 输电损耗率 (跨城)
    transmission_loss_rate: float = 0.05

@dataclass
class StateVector:
    """状态向量：描述城市的当前状态（随时间快速变化）"""
    grid_load: float = 0.0                 # 当前电网负荷 (MW)
    renewable_generation: float = 0.0      # 当前可再生能源发电 (MW)
    battery_soc: float = 0.5                # 储能荷电状态 (0-1)
    computing_served: float = 0.0           # 已服务的算力需求 (PFLOPS)
    temperature: float = 20.0                # 当前温度 (摄氏度)
    irradiance: float = 0.0                  # 当前太阳辐照 (kW/m2)
    wind_speed: float = 0.0                  # 当前风速 (m/s)

# ===================== 抽象基类（定义扩展接口）=====================

class NaturalModel(abc.ABC):
    """自然条件模型接口：根据时间和位置计算太阳、风、温度"""
    @abc.abstractmethod
    def get_forces(self, time_hours: int, location: Tuple[float, float]) -> Dict[str, float]:
        """返回包含 'irradiance', 'wind_speed', 'temperature' 的字典"""
        pass

class InfrastructureModel(abc.ABC):
    """基础设施动态模型接口：描述城市内部系统动力学"""
    @abc.abstractmethod
    def evolve(self, 
               state: StateVector, 
               capabilities: CapabilityVector,
               natural_forces: Dict[str, float],
               dt_hours: float) -> StateVector:
        """根据当前状态、能力和自然力，更新状态"""
        pass

class Policy(abc.ABC):
    """政策干预接口：描述一个政策如何影响城市"""
    def __init__(self, 
                 policy_id: str,
                 policy_type: 'PolicyType',
                 start_time: int,
                 end_time: int,
                 target_cities: List[str],
                 intensity: float = 1.0):
        self.id = policy_id
        self.type = policy_type
        self.start = start_time
        self.end = end_time
        self.target_cities = set(target_cities)
        self.intensity = intensity  # 政策强度系数

    @abc.abstractmethod
    def apply(self, city: 'CityNode', current_time: int) -> None:
        """应用政策到城市——子类实现具体逻辑"""
        pass

    def is_active(self, city_id: str, current_time: int) -> bool:
        return (city_id in self.target_cities and 
                self.start <= current_time < self.end)

class InteractionRule(abc.ABC):
    """城市间交互规则接口：定义两个城市如何交换资源"""
    @abc.abstractmethod
    def interact(self, city_a: 'CityNode', city_b: 'CityNode', dt_hours: float) -> None:
        pass

class DataConnector(abc.ABC):
    """外部数据接入接口：从真实世界获取参数用于初始化或校准"""
    @abc.abstractmethod
    def get_city_capabilities(self, city_id: str) -> CapabilityVector:
        pass

    @abc.abstractmethod
    def get_city_state(self, city_id: str, time: int) -> Optional[StateVector]:
        pass

    @abc.abstractmethod
    def get_policy_parameters(self, policy_id: str) -> Dict[str, Any]:
        pass

# ===================== 枚举类型 =====================

class PolicyType(Enum):
    INVESTMENT = "investment"      # 投入型（补贴、税收优惠）
    REGULATORY = "regulatory"       # 规制型（标准、限额）
    INFORMATION = "information"     # 信息型（披露、标签）
    INFRASTRUCTURE = "infrastructure" # 基建型（网络、电网）

# ===================== 默认模型实现 =====================

class SimpleNaturalModel(NaturalModel):
    """简化的自然模型：基于正弦曲线模拟日照和温度，风速随机"""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def get_forces(self, time_hours: int, location: Tuple[float, float]) -> Dict[str, float]:
        hour = time_hours % 24
        day = (time_hours // 24) % 365

        # 日照：白天正弦，夜晚为零
        if 6 <= hour <= 18:
            irradiance = np.sin(np.pi * (hour - 6) / 12)
        else:
            irradiance = 0.0

        # 温度：基础15度，白天高，晚上低，加上季节变化
        base_temp = 15 + 10 * np.sin(2 * np.pi * (day - 80) / 365)
        daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        temperature = base_temp + daily_variation

        # 风速：随机游走简化
        wind_speed = 5 + 3 * self.rng.randn()
        wind_speed = max(0, wind_speed)

        return {'irradiance': irradiance, 'wind_speed': wind_speed, 'temperature': temperature}

class SimpleInfrastructureModel(InfrastructureModel):
    """简化的基础设施动态模型"""
    def evolve(self, state: StateVector, capabilities: CapabilityVector,
               natural_forces: Dict[str, float], dt_hours: float) -> StateVector:
        irradiance = natural_forces.get('irradiance', 0)
        wind_speed = natural_forces.get('wind_speed', 0)
        temperature = natural_forces.get('temperature', 20)

        # 可再生能源发电
        solar_power = capabilities.solar_capacity * irradiance * 0.2
        wind_power = capabilities.wind_capacity * (wind_speed / 12)
        renewable = solar_power + wind_power

        # 负荷：基础负荷 + 温度制冷负荷
        base_load = 0.6 * capabilities.grid_capacity
        cooling_load = base_load * 0.1 * max(0, temperature - 20)
        total_load = base_load + cooling_load

        # 储能更新
        net_load = total_load - renewable
        battery_change = -net_load * dt_hours / capabilities.battery_capacity if capabilities.battery_capacity > 0 else 0
        new_soc = state.battery_soc + battery_change
        new_soc = np.clip(new_soc, 0, 1)

        if new_soc < 0:
            total_load += new_soc * capabilities.battery_capacity / dt_hours
            new_soc = 0
        elif new_soc > 1:
            renewable -= (new_soc - 1) * capabilities.battery_capacity / dt_hours
            new_soc = 1

        return StateVector(
            grid_load=total_load,
            renewable_generation=renewable,
            battery_soc=new_soc,
            computing_served=state.computing_served,
            temperature=temperature,
            irradiance=irradiance,
            wind_speed=wind_speed
        )

# ===================== 政策子类实现 =====================

class InvestmentPolicy(Policy):
    """投入型政策：补贴、税收优惠、资金支持"""
    def __init__(self, 
                 policy_id: str,
                 start_time: int,
                 end_time: int,
                 target_cities: List[str],
                 target_capability: str,      # 要增强的能力
                 increase_rate: float,         # 增长率（如0.2）
                 intensity: float = 1.0):
        super().__init__(policy_id, PolicyType.INVESTMENT, start_time, end_time, target_cities, intensity)
        self.target_capability = target_capability
        self.increase_rate = increase_rate * intensity

    def apply(self, city: 'CityNode', current_time: int) -> None:
        if not self.is_active(city.id, current_time):
            return
        if hasattr(city.capabilities, self.target_capability):
            old = getattr(city.capabilities, self.target_capability)
            new = old * (1 + self.increase_rate)
            setattr(city.capabilities, self.target_capability, new)

class RegulatoryPolicy(Policy):
    """规制型政策：能效标准、碳排放限额"""
    def __init__(self,
                 policy_id: str,
                 start_time: int,
                 end_time: int,
                 target_cities: List[str],
                 constraint_type: str,         # 'max_emission', 'min_efficiency'
                 threshold: float,
                 penalty_factor: float = 0.1,
                 intensity: float = 1.0):
        super().__init__(policy_id, PolicyType.REGULATORY, start_time, end_time, target_cities, intensity)
        self.constraint_type = constraint_type
        self.threshold = threshold
        self.penalty = penalty_factor * intensity

    def apply(self, city: 'CityNode', current_time: int) -> None:
        if not self.is_active(city.id, current_time):
            return

        if self.constraint_type == 'max_emission':
            # 碳排放约束
            emission = (city.state.grid_load - city.state.renewable_generation) * city.capabilities.emission_factor
            if emission > self.threshold:
                excess = emission - self.threshold
                penalty_load = excess * self.penalty
                city.state.grid_load += penalty_load

        elif self.constraint_type == 'min_efficiency':
            # 能效约束（值越小效率越高）
            if city.capabilities.computing_efficiency > self.threshold:
                # 惩罚：降低算力服务
                reduction = (city.capabilities.computing_efficiency - self.threshold) / city.capabilities.computing_efficiency
                city.state.computing_served *= (1 - reduction * self.penalty)

class InformationPolicy(Policy):
    """信息型政策：碳标签、能效标识"""
    def __init__(self,
                 policy_id: str,
                 start_time: int,
                 end_time: int,
                 target_cities: List[str],
                 disclosure_type: str,        # 'carbon_label', 'efficiency_grade'
                 consumer_sensitivity: float = 0.1,
                 intensity: float = 1.0):
        super().__init__(policy_id, PolicyType.INFORMATION, start_time, end_time, target_cities, intensity)
        self.disclosure_type = disclosure_type
        self.sensitivity = consumer_sensitivity * intensity

    def apply(self, city: 'CityNode', current_time: int) -> None:
        if not self.is_active(city.id, current_time):
            return

        if self.disclosure_type == 'carbon_label':
            # 高碳算力需求下降
            carbon_intensity = (city.state.grid_load - city.state.renewable_generation) * city.capabilities.emission_factor / (city.state.computing_served + 1e-6)
            demand_reduction = carbon_intensity * self.sensitivity
            city.state.computing_served *= max(0, 1 - demand_reduction)

class InfrastructurePolicy(Policy):
    """基建型政策：算力网络、能源设施"""
    def __init__(self,
                 policy_id: str,
                 start_time: int,
                 end_time: int,
                 target_cities: List[str],
                 infrastructure_type: str,     # 'grid_connection', 'network_bandwidth'
                 capacity_increase: float,
                 intensity: float = 1.0):
        super().__init__(policy_id, PolicyType.INFRASTRUCTURE, start_time, end_time, target_cities, intensity)
        self.infrastructure_type = infrastructure_type
        self.capacity_increase = capacity_increase * intensity

    def apply(self, city: 'CityNode', current_time: int) -> None:
        if not self.is_active(city.id, current_time):
            return

        if self.infrastructure_type == 'grid_connection':
            city.capabilities.transmission_loss_rate *= (1 - self.capacity_increase)
        elif self.infrastructure_type == 'network_bandwidth':
            city.capabilities.network_bandwidth *= (1 + self.capacity_increase)

# ===================== 政策辅助类（执行差距、时滞、自适应）=====================

class PolicyImplementationGap:
    """政策执行差距模型"""
    def __init__(self, design_intensity: float, implementation_rate: float):
        self.design = design_intensity
        self.rate = implementation_rate

    @property
    def actual_intensity(self) -> float:
        return self.design * self.rate

    def apply_to_policy(self, policy: Policy) -> Policy:
        """返回一个带有执行差距的政策副本（仅修改强度）"""
        new_policy = copy.deepcopy(policy)
        new_policy.intensity = self.actual_intensity
        return new_policy

class PolicyWithLag(Policy):
    """带有时滞效应的政策装饰器"""
    def __init__(self, base_policy: Policy, lag_hours: int):
        self.base = base_policy
        self.lag = lag_hours
        # 继承基本属性以便is_active使用
        super().__init__(base_policy.id, base_policy.type, 
                         base_policy.start + lag_hours, 
                         base_policy.end + lag_hours, 
                         list(base_policy.target_cities), 
                         base_policy.intensity)

    def apply(self, city: 'CityNode', current_time: int) -> None:
        # 实际生效时间已由父类start控制
        self.base.apply(city, current_time)

class AdaptivePolicy(Policy):
    """自适应政策——根据城市状态动态调整强度"""
    def __init__(self,
                 base_policy: Policy,
                 adaptation_func: Callable[['CityNode', int], float],
                 min_intensity: float = 0.0,
                 max_intensity: float = 2.0):
        self.base = base_policy
        self.adapt_func = adaptation_func
        self.min_i = min_intensity
        self.max_i = max_intensity
        # 复制基础属性
        super().__init__(base_policy.id, base_policy.type, 
                         base_policy.start, base_policy.end, 
                         list(base_policy.target_cities), base_policy.intensity)

    def apply(self, city: 'CityNode', current_time: int) -> None:
        if not self.is_active(city.id, current_time):
            return
        # 计算动态强度
        raw = self.adapt_func(city, current_time)
        self.base.intensity = np.clip(raw, self.min_i, self.max_i)
        self.base.apply(city, current_time)

class PolicyMix:
    """政策组合——同时应用多个政策"""
    def __init__(self, policies: List[Policy]):
        self.policies = policies

    def apply_all(self, city: 'CityNode', current_time: int) -> None:
        for p in self.policies:
            p.apply(city, current_time)

# ===================== 核心城市节点 =====================

class CityNode:
    """城市点位智能体"""
    def __init__(self,
                 city_id: str,
                 location: Tuple[float, float],
                 capabilities: CapabilityVector,
                 initial_state: Optional[StateVector] = None,
                 natural_model: Optional[NaturalModel] = None,
                 infra_model: Optional[InfrastructureModel] = None):
        self.id = city_id
        self.location = location
        self.capabilities = capabilities
        self.state = initial_state or StateVector()
        self.natural_model = natural_model or SimpleNaturalModel()
        self.infra_model = infra_model or SimpleInfrastructureModel()

        self.history: List[Tuple[int, StateVector]] = []
        self.policy_history: List[Dict] = []  # 记录政策干预

    def step(self, current_time: int, dt_hours: float = 1.0, policies: List[Policy] = None) -> None:
        # 应用政策
        if policies:
            for p in policies:
                p.apply(self, current_time)

        # 获取自然力
        natural = self.natural_model.get_forces(current_time, self.location)

        # 内部动态演化
        self.state = self.infra_model.evolve(self.state, self.capabilities, natural, dt_hours)

        # 记录历史
        self.history.append((current_time, copy.deepcopy(self.state)))

    def get_metrics(self) -> Dict[str, float]:
        """输出当前宏观指标"""
        # 算力利用率
        computing_util = self.state.computing_served / (self.capabilities.computing_power + 1e-6)
        # 可再生能源占比
        renewable_ratio = self.state.renewable_generation / (self.state.grid_load + 1e-6)
        # 碳强度
        carbon_intensity = (self.state.grid_load - self.state.renewable_generation) * self.capabilities.emission_factor / (self.state.computing_served + 1e-6)
        # 电网稳定性（简化：负荷与容量比）
        grid_stability = 1 - abs(self.state.grid_load / self.capabilities.grid_capacity - 0.5) * 2

        return {
            'computing_util': computing_util,
            'renewable_ratio': renewable_ratio,
            'carbon_intensity': carbon_intensity,
            'grid_stability': grid_stability,
            'battery_soc': self.state.battery_soc,
            'grid_load': self.state.grid_load,
            'renewable_gen': self.state.renewable_generation,
            'computing_served': self.state.computing_served
        }

# ===================== 网络与交互 =====================

class SimpleInteractionRule(InteractionRule):
    """简单的资源交互规则：电力传输和算力迁移"""
    def interact(self, city_a: 'CityNode', city_b: 'CityNode', dt_hours: float) -> None:
        # 电力传输：从盈余到缺电
        surplus_a = max(0, city_a.state.renewable_generation - city_a.state.grid_load)
        deficit_b = max(0, city_b.state.grid_load - city_b.state.renewable_generation)
        # 只有当储能接近极限时才传输（简化）
        if city_a.state.battery_soc > 0.95 and deficit_b > 0 and surplus_a > 0:
            transfer = min(surplus_a, deficit_b)
            loss = transfer * city_a.capabilities.transmission_loss_rate
            city_a.state.renewable_generation -= transfer
            city_b.state.renewable_generation += (transfer - loss)

        # 算力迁移：A绿电充裕且B需求未满足
        if (city_a.state.renewable_generation > 1.2 * city_a.state.grid_load and
            city_b.state.computing_served < city_b.capabilities.computing_power * 0.9):
            capacity_a = (city_a.state.renewable_generation - city_a.state.grid_load) / city_a.capabilities.computing_efficiency
            demand_b = city_b.capabilities.computing_power * 0.9 - city_b.state.computing_served
            transfer = min(capacity_a, demand_b)
            if transfer > 0:
                extra_load = transfer * city_a.capabilities.computing_efficiency
                city_a.state.grid_load += extra_load
                city_a.state.renewable_generation -= extra_load
                city_b.state.computing_served += transfer

class NetworkManager:
    """管理城市间的连接和交互"""
    def __init__(self, city_nodes: List[CityNode], connection_threshold_km: float = 500):
        self.cities = {c.id: c for c in city_nodes}
        self.connections = self._build_proximity_network(connection_threshold_km)
        self.interaction_rules: List[InteractionRule] = [SimpleInteractionRule()]

    def _build_proximity_network(self, threshold_km: float) -> Dict[str, List[str]]:
        import geopy.distance
        conn = {}
        ids = list(self.cities.keys())
        for i, cid1 in enumerate(ids):
            for cid2 in ids[i+1:]:
                c1 = self.cities[cid1]
                c2 = self.cities[cid2]
                dist = geopy.distance.distance(c1.location, c2.location).km
                if dist < threshold_km:
                    conn.setdefault(cid1, []).append(cid2)
                    conn.setdefault(cid2, []).append(cid1)
        return conn

    def get_neighbors(self, city_id: str) -> List[CityNode]:
        return [self.cities[nid] for nid in self.connections.get(city_id, [])]

    def apply_interactions(self, dt_hours: float) -> None:
        for cid, neighbors in self.connections.items():
            city = self.cities[cid]
            for nid in neighbors:
                if cid < nid:  # 避免重复
                    other = self.cities[nid]
                    for rule in self.interaction_rules:
                        rule.interact(city, other, dt_hours)

# ===================== 指标收集器 =====================

class MetricsCollector:
    def __init__(self):
        self.data = []

    def record(self, time: int, cities: List[CityNode]):
        for city in cities:
            metrics = city.get_metrics()
            record = {'time': time, 'city_id': city.id, **metrics}
            self.data.append(record)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    def reset(self):
        self.data = []

# ===================== 仿真引擎 =====================

class SimulationEngine:
    def __init__(self,
                 cities: List[CityNode],
                 policies: Optional[List[Policy]] = None,
                 network_manager: Optional[NetworkManager] = None):
        self.cities = cities
        self.policies = policies or []
        self.network = network_manager or NetworkManager(cities)
        self.metrics = MetricsCollector()
        self.current_time = 0  # 小时

    def run(self, duration_hours: int, dt: float = 1.0) -> pd.DataFrame:
        self.metrics.reset()
        for _ in range(duration_hours):
            # 城市内部演化（同时应用政策）
            for city in self.cities:
                active_policies = [p for p in self.policies if p.is_active(city.id, self.current_time)]
                city.step(self.current_time, dt, active_policies)

            # 城市间交互
            self.network.apply_interactions(dt)

            # 记录指标
            self.metrics.record(self.current_time, self.cities)

            self.current_time += dt

        return self.metrics.get_dataframe()

    def reset(self):
        self.current_time = 0
        for city in self.cities:
            city.history.clear()
            city.policy_history.clear()
        self.metrics.reset()

# ===================== 外部数据接入示例 =====================

class MockDataConnector(DataConnector):
    """模拟数据连接器（用于测试）"""
    def __init__(self):
        self.city_data = {
            'BJ': {
                'capabilities': CapabilityVector(
                    solar_capacity=100, wind_capacity=50, grid_capacity=10000,
                    battery_capacity=500, computing_power=500, computing_efficiency=0.5,
                    emission_factor=0.8, network_bandwidth=1000
                ),
                'state': StateVector(grid_load=8000, renewable_generation=80, battery_soc=0.5, computing_served=400)
            },
            'GZ': {
                'capabilities': CapabilityVector(
                    solar_capacity=300, wind_capacity=200, grid_capacity=5000,
                    battery_capacity=1000, computing_power=200, computing_efficiency=0.3,
                    emission_factor=0.2, network_bandwidth=500
                ),
                'state': StateVector(grid_load=3000, renewable_generation=400, battery_soc=0.8, computing_served=150)
            }
        }

    def get_city_capabilities(self, city_id: str) -> CapabilityVector:
        return self.city_data[city_id]['capabilities']

    def get_city_state(self, city_id: str, time: int) -> Optional[StateVector]:
        # 简化：忽略时间，返回初始状态
        return self.city_data[city_id]['state']

    def get_policy_parameters(self, policy_id: str) -> Dict[str, Any]:
        return {}

# ===================== 政策优化器（高级）=====================

class PolicyOptimizer:
    """基于仿真的政策优化器（简单随机搜索示例）"""
    def __init__(self,
                 engine: SimulationEngine,
                 policy_space: Dict[str, List[Any]],
                 objective_func: Callable[[pd.DataFrame], float]):
        self.engine = engine
        self.policy_space = policy_space
        self.objective = objective_func

    def random_search(self, n_trials: int = 100) -> Dict[str, Any]:
        best_score = -np.inf
        best_params = None

        for _ in range(n_trials):
            # 随机采样政策参数
            params = {
                'type': np.random.choice(self.policy_space['type']),
                'intensity': np.random.uniform(*self.policy_space['intensity_range']),
                'target': np.random.choice(self.policy_space['targets']),
                'start': np.random.randint(*self.policy_space['start_range']),
                'duration': np.random.randint(*self.policy_space['duration_range'])
            }
            # 创建政策（简化：只创建一种政策用于演示）
            policy = self._create_policy(params)
            self.engine.policies = [policy]
            self.engine.reset()
            results = self.engine.run(duration_hours=self.policy_space['sim_duration'])
            score = self.objective(results)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params

    def _create_policy(self, params: Dict) -> Policy:
        # 根据类型创建政策实例
        if params['type'] == 'investment':
            return InvestmentPolicy(
                policy_id='opt_policy',
                start_time=params['start'],
                end_time=params['start'] + params['duration'],
                target_cities=[params['target']],
                target_capability='computing_efficiency',
                increase_rate=0.2,
                intensity=params['intensity']
            )
        # 其他类型可扩展
        else:
            raise NotImplementedError

# ===================== 运行示例 =====================

if __name__ == "__main__":
    # 1. 初始化城市（使用模拟数据）
    connector = MockDataConnector()
    beijing = CityNode('BJ', (39.9, 116.4),
                       connector.get_city_capabilities('BJ'),
                       connector.get_city_state('BJ', 0))
    guizhou = CityNode('GZ', (26.5, 106.7),
                       connector.get_city_capabilities('GZ'),
                       connector.get_city_state('GZ', 0))
    cities = [beijing, guizhou]

    # 2. 定义政策组合（考虑执行差距）
    policies = [
        InvestmentPolicy('subsidy_gz', start_time=24*30, end_time=24*365,
                         target_cities=['GZ'], target_capability='computing_efficiency',
                         increase_rate=0.2, intensity=1.0),
        RegulatoryPolicy('eff_std_bj', start_time=24*30, end_time=24*365,
                         target_cities=['BJ'], constraint_type='min_efficiency',
                         threshold=0.45, penalty_factor=0.1, intensity=1.2),
        InfrastructurePolicy('network_expand', start_time=24*60, end_time=24*365,
                             target_cities=['BJ', 'GZ'], infrastructure_type='network_bandwidth',
                             capacity_increase=0.3, intensity=1.0)
    ]

    # 应用执行差距（假设西部执行率90%，东部95%）
    gaps = {'GZ': 0.9, 'BJ': 0.95}
    adjusted_policies = []
    for p in policies:
        for city_id in p.target_cities:
            if city_id in gaps:
                p_copy = copy.deepcopy(p)
                p_copy.intensity *= gaps[city_id]
                adjusted_policies.append(p_copy)
            else:
                adjusted_policies.append(p)

    # 3. 运行基线情景（无政策）
    engine_baseline = SimulationEngine(cities=[copy.deepcopy(c) for c in cities], policies=[])
    results_baseline = engine_baseline.run(duration_hours=24*180)  # 半年

    # 4. 运行政策情景
    engine_policy = SimulationEngine(cities=[copy.deepcopy(c) for c in cities], policies=adjusted_policies)
    results_policy = engine_policy.run(duration_hours=24*180)

    # 5. 评估效果
    def summarize(df, city):
        subset = df[df.city_id == city]
        return {
            'computing_util': subset['computing_util'].mean(),
            'renewable_ratio': subset['renewable_ratio'].mean(),
            'carbon_intensity': subset['carbon_intensity'].mean(),
            'grid_stability': subset['grid_stability'].mean()
        }

    print("="*60)
    print("政策效果评估报告")
    print("="*60)
    for city in ['BJ', 'GZ']:
        print(f"\n城市 {city}:")
        base = summarize(results_baseline, city)
        pol = summarize(results_policy, city)
        for metric in base.keys():
            change = (pol[metric] - base[metric]) / base[metric] * 100
            arrow = "↑" if change > 0 else "↓"
            print(f"  {metric:20}: {base[metric]:.3f} → {pol[metric]:.3f} {arrow} {abs(change):.1f}%")

    # 可选：政策优化搜索
    # optimizer = PolicyOptimizer(engine_baseline, {...}, objective_func=lambda df: df['computing_util'].mean())
    # best = optimizer.random_search(n_trials=10)
    # print("最优政策参数:", best)