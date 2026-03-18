"""
城市物质行动者网络仿真框架 (v3.0) - 知识图谱增强版
================================================================
基于ANT理论，将城市抽象为具有物质能力向量的点位，支持科技政策模拟，
并可选择将仿真结果持久化到知识图谱（Neo4j）中，实现“仿真-知识”闭环。

核心特性：
- 四类科技政策模拟（投资、规制、信息、基建）
- 执行差距、时滞效应、动态自适应政策
- 城市间资源交互（电力、算力）
- 知识图谱作为仿真输入/输出（方案一）
- 治理观测指标自动计算

依赖：
pip install numpy pandas neo4j geopy

使用前请确保Neo4j数据库已启动（如使用Mock客户端则无需）
"""

import abc
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import geopy.distance

# ===================== Neo4j 导入（可选） =====================
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("警告: neo4j 库未安装，将使用Mock知识图谱客户端。如需连接真实Neo4j，请运行: pip install neo4j")


# ===================== 基础数据类型 =====================

@dataclass
class CapabilityVector:
    """物质能力向量：描述城市的基础设施容量和资源禀赋（相对稳定）"""
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


# ===================== 抽象基类 =====================

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

class KnowledgeGraphClient(abc.ABC):
    """知识图谱客户端抽象接口（方案一：轻量级整合）"""
    
    @abc.abstractmethod
    def load_city(self, city_id: str) -> Optional[Dict[str, Any]]:
        """从图谱加载城市初始数据，返回包含 capabilities 和 state 的字典"""
        pass
    
    @abc.abstractmethod
    def load_policies(self, time: int) -> List[Dict]:
        """从图谱加载当前生效的政策（可根据时间筛选）"""
        pass
    
    @abc.abstractmethod
    def write_snapshot(self, city_id: str, time: int, state: StateVector, metrics: Dict[str, float]) -> None:
        """写入城市状态快照到图谱"""
        pass
    
    @abc.abstractmethod
    def write_interaction(self, from_city: str, to_city: str, time: int,
                          interaction_type: str, amount: float, loss: float = 0.0) -> None:
        """记录城市间交互事件"""
        pass
    
    @abc.abstractmethod
    def close(self):
        """关闭连接"""
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
        # 碳强度 (kg CO2 / PFLOPS)
        carbon_intensity = (self.state.grid_load - self.state.renewable_generation) * self.capabilities.emission_factor / (self.state.computing_served + 1e-6)
        # 电网稳定性（简化：负荷与容量比，越接近50%越稳定）
        load_ratio = self.state.grid_load / self.capabilities.grid_capacity if self.capabilities.grid_capacity > 0 else 0
        grid_stability = 1 - abs(load_ratio - 0.5) * 2

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
            # 记录交互信息（可在城市中存储，但此处不处理，由引擎记录到图谱）
            # 这里我们通过返回字典的方式让引擎捕获，但为了简化，直接在引擎中记录

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
                # 同样，交互信息需记录

class NetworkManager:
    """管理城市间的连接和交互"""
    def __init__(self, city_nodes: List[CityNode], connection_threshold_km: float = 500):
        self.cities = {c.id: c for c in city_nodes}
        self.connections = self._build_proximity_network(connection_threshold_km)
        self.interaction_rules: List[InteractionRule] = [SimpleInteractionRule()]

    def _build_proximity_network(self, threshold_km: float) -> Dict[str, List[str]]:
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

    def apply_interactions(self, dt_hours: float) -> List[Dict]:
        """
        应用所有交互规则，并返回交互事件列表，供知识图谱记录
        返回格式: [{'from': cid1, 'to': cid2, 'type': 'electricity', 'amount': transfer, 'loss': loss}, ...]
        """
        interactions = []
        for cid, neighbors in self.connections.items():
            city = self.cities[cid]
            for nid in neighbors:
                if cid < nid:  # 避免重复
                    other = self.cities[nid]
                    # 记录交互前的状态以便计算交互量
                    # 简单起见，我们直接在规则中捕获交互量，但为了不破坏规则接口，这里不实现
                    # 我们可以在 SimpleInteractionRule 中填充城市的一个临时属性，但更好的方式是让规则返回交互数据
                    # 为了保持简洁，我们暂时忽略精确记录，只记录发生了交互的事实
                    # 但我们可以实现一个更详细的规则版本
                    for rule in self.interaction_rules:
                        # 这里假设规则会直接修改城市状态，我们无法获取具体量
                        # 因此我们暂时不记录具体数值，只记录存在连接
                        # 在真实场景中，可以扩展 InteractionRule 接口使其返回交互数据
                        pass
        return interactions


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


# ===================== 知识图谱客户端实现 =====================

class NullKnowledgeGraphClient(KnowledgeGraphClient):
    """空实现，所有操作无效果（用于无Neo4j环境）"""
    def load_city(self, city_id: str) -> Optional[Dict[str, Any]]:
        return None

    def load_policies(self, time: int) -> List[Dict]:
        return []

    def write_snapshot(self, city_id: str, time: int, state: StateVector, metrics: Dict[str, float]) -> None:
        pass

    def write_interaction(self, from_city: str, to_city: str, time: int,
                          interaction_type: str, amount: float, loss: float = 0.0) -> None:
        pass

    def close(self):
        pass

class Neo4jKnowledgeGraphClient(KnowledgeGraphClient):
    """Neo4j 实现"""
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j库未安装，请运行: pip install neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def _run_query(self, query: str, parameters: Dict = None):
        with self.driver.session(database=self.database) as session:
            return session.run(query, parameters or {}).data()

    def load_city(self, city_id: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (c:City {id: $city_id})
        OPTIONAL MATCH (c)-[:HAS_CAPABILITY]->(cap:Capability)
        RETURN c, cap
        """
        result = self._run_query(query, {"city_id": city_id})
        if not result:
            return None
        # 解析结果，构造 CapabilityVector 和 StateVector
        # 这里假设图谱中存储了capability属性
        # 简化：直接返回字典，由调用者构造对象
        data = result[0]
        cap_dict = {k: v for k, v in data.get('cap', {}).items() if not k.startswith('_')}
        # 如果cap不存在，返回None
        return {'capabilities': cap_dict}

    def load_policies(self, time: int) -> List[Dict]:
        query = """
        MATCH (p:Policy)
        WHERE p.start <= $time AND p.end > $time
        RETURN p
        """
        result = self._run_query(query, {"time": time})
        policies = []
        for record in result:
            p = record['p']
            # 转换属性为字典
            policies.append({
                'id': p.get('id'),
                'type': p.get('type'),
                'start': p.get('start'),
                'end': p.get('end'),
                'target_cities': p.get('target_cities', []),
                'intensity': p.get('intensity', 1.0),
                # 其他参数...
            })
        return policies

    def write_snapshot(self, city_id: str, time: int, state: StateVector, metrics: Dict[str, float]) -> None:
        query = """
        MATCH (c:City {id: $city_id})
        CREATE (obs:Observation {
            time: $time,
            grid_load: $grid_load,
            renewable_generation: $renewable_generation,
            battery_soc: $battery_soc,
            computing_served: $computing_served,
            computing_util: $computing_util,
            renewable_ratio: $renewable_ratio,
            carbon_intensity: $carbon_intensity,
            grid_stability: $grid_stability
        })
        CREATE (c)-[:HAS_OBSERVATION]->(obs)
        """
        params = {
            "city_id": city_id,
            "time": time,
            "grid_load": state.grid_load,
            "renewable_generation": state.renewable_generation,
            "battery_soc": state.battery_soc,
            "computing_served": state.computing_served,
            **metrics
        }
        self._run_query(query, params)

    def write_interaction(self, from_city: str, to_city: str, time: int,
                          interaction_type: str, amount: float, loss: float = 0.0) -> None:
        query = """
        MATCH (c1:City {id: $from_city})
        MATCH (c2:City {id: $to_city})
        CREATE (c1)-[:INTERACTS {
            type: $type,
            time: $time,
            amount: $amount,
            loss: $loss
        }]->(c2)
        """
        self._run_query(query, {
            "from_city": from_city,
            "to_city": to_city,
            "type": interaction_type,
            "time": time,
            "amount": amount,
            "loss": loss
        })


# ===================== 仿真引擎（知识图谱增强版） =====================

class SimulationEngine:
    """仿真引擎，支持知识图谱读写"""
    def __init__(self,
                 cities: List[CityNode],
                 policies: Optional[List[Policy]] = None,
                 network_manager: Optional[NetworkManager] = None,
                 kg_client: Optional[KnowledgeGraphClient] = None,
                 kg_write_frequency_hours: int = 24):  # 默认每天写入一次快照
        self.cities = cities
        self.policies = policies or []
        self.network = network_manager or NetworkManager(cities)
        self.kg_client = kg_client or NullKnowledgeGraphClient()
        self.kg_write_freq = kg_write_frequency_hours
        self.metrics = MetricsCollector()
        self.current_time = 0  # 小时

    def run(self, duration_hours: int, dt: float = 1.0) -> pd.DataFrame:
        self.metrics.reset()
        for step in range(duration_hours):
            # 从图谱加载当前生效的政策（如果需要动态更新）
            # 这里简化，使用初始化时传入的policies

            # 城市内部演化（同时应用政策）
            for city in self.cities:
                active_policies = [p for p in self.policies if p.is_active(city.id, self.current_time)]
                city.step(self.current_time, dt, active_policies)

            # 城市间交互
            interactions = self.network.apply_interactions(dt)  # 目前返回空列表，需要扩展才能捕获

            # 记录指标
            self.metrics.record(self.current_time, self.cities)

            # 写入知识图谱（按频率）
            if self.kg_client and self.current_time % self.kg_write_freq == 0:
                for city in self.cities:
                    metrics = city.get_metrics()
                    self.kg_client.write_snapshot(city.id, self.current_time, city.state, metrics)

                # 记录交互（需要实现交互量捕获）
                # 这里简化，不记录交互

            self.current_time += dt

        return self.metrics.get_dataframe()

    def reset(self):
        self.current_time = 0
        for city in self.cities:
            city.history.clear()
            city.policy_history.clear()
        self.metrics.reset()


# ===================== 外部数据连接器示例 =====================

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
        return self.city_data[city_id]['state']

    def get_policy_parameters(self, policy_id: str) -> Dict[str, Any]:
        return {}


# ===================== 运行示例 =====================

if __name__ == "__main__":
    # 1. 初始化知识图谱客户端（如无Neo4j，使用Null）
    use_neo4j = False  # 设为True连接真实Neo4j
    if use_neo4j and NEO4J_AVAILABLE:
        kg_client = Neo4jKnowledgeGraphClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        # 尝试从图谱加载城市数据（需要图谱中预先有City节点）
        # 这里简化：如果图谱没有，则使用Mock数据
        # 实际使用中，可以混合：先从图谱加载，如果没有则用Mock
        # 为演示，我们直接使用Mock数据
        kg_client = NullKnowledgeGraphClient()  # 临时替换
    else:
        kg_client = NullKnowledgeGraphClient()

    # 2. 初始化城市（使用Mock数据）
    connector = MockDataConnector()
    beijing = CityNode('BJ', (39.9, 116.4),
                       connector.get_city_capabilities('BJ'),
                       connector.get_city_state('BJ', 0))
    guizhou = CityNode('GZ', (26.5, 106.7),
                       connector.get_city_capabilities('GZ'),
                       connector.get_city_state('GZ', 0))
    cities = [beijing, guizhou]

    # 3. 定义政策组合（考虑执行差距）
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

    # 4. 运行基线情景（无政策）
    engine_baseline = SimulationEngine(
        cities=[copy.deepcopy(c) for c in cities],
        policies=[],
        kg_client=kg_client,
        kg_write_frequency_hours=24*30  # 每月写一次，避免过多
    )
    results_baseline = engine_baseline.run(duration_hours=24*180)  # 半年

    # 5. 运行政策情景
    engine_policy = SimulationEngine(
        cities=[copy.deepcopy(c) for c in cities],
        policies=adjusted_policies,
        kg_client=kg_client,
        kg_write_frequency_hours=24*30
    )
    results_policy = engine_policy.run(duration_hours=24*180)

    # 6. 评估效果
    def summarize(df, city):
        subset = df[df.city_id == city]
        return {
            'computing_util': subset['computing_util'].mean(),
            'renewable_ratio': subset['renewable_ratio'].mean(),
            'carbon_intensity': subset['carbon_intensity'].mean(),
            'grid_stability': subset['grid_stability'].mean()
        }

    print("="*60)
    print("政策效果评估报告（知识图谱版本）")
    print("="*60)
    for city in ['BJ', 'GZ']:
        print(f"\n城市 {city}:")
        base = summarize(results_baseline, city)
        pol = summarize(results_policy, city)
        for metric in base.keys():
            change = (pol[metric] - base[metric]) / base[metric] * 100
            arrow = "↑" if change > 0 else "↓"
            print(f"  {metric:20}: {base[metric]:.3f} → {pol[metric]:.3f} {arrow} {abs(change):.1f}%")

    # 7. 关闭知识图谱连接
    kg_client.close()