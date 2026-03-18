# 城市物质行动者网络仿真框架 (Urban Material Actant Network Simulation Framework)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Neo4j Integration](https://img.shields.io/badge/Neo4j-ready-green)

一个基于**行动者网络理论（Actor-Network Theory, ANT）** 的多智能体仿真框架，将城市抽象为具有物质能力向量的点位，用于科技政策的模拟与评估。支持投资、规制、信息、基建四类政策，并可选择与**Neo4j 知识图谱**集成，实现“仿真-知识”闭环。

---

## 📖 项目简介

在人工智能与“双碳”目标双重背景下，科技政策的制定亟需可量化、可推演的工具。本框架将**城市**视为具有物质能力的行动者（如电网容量、算力规模、可再生能源禀赋），通过多智能体建模模拟城市间资源流动与政策干预效果。核心创新：

- **ANT 广义对称性**：城市、自然条件、基础设施均为平等行动者，通过状态变化交互。
- **有限复杂抽象**：以城市为最小分析单元，内部用系统动力学聚合，避免微观细节爆炸。
- **政策可计算化**：将政策转化为可参数化、可组合、可动态调整的代码模块。
- **知识图谱集成**：将仿真结果持久化到 Neo4j，支持复杂查询与反事实推理。

---

## ✨ 特性

- 🏙️ **城市点位建模**：每个城市封装为 `CityNode`，包含物质能力（`CapabilityVector`）和动态状态（`StateVector`）。
- 🔌 **可插拔模型**：自然条件、基础设施演化、城市交互规则均基于抽象接口，可自由替换。
- 📊 **四类政策模拟**：
  - **投资型**：补贴、税收优惠，增强特定能力（如 `computing_efficiency`）。
  - **规制型**：能效标准、碳排放限额，附带惩罚机制。
  - **信息型**：碳标签、能效标识，影响需求侧行为。
  - **基建型**：电网互联、网络扩容，降低传输损耗。
- ⏱️ **政策执行细节**：支持执行差距、时间滞后、动态自适应强度。
- 🔗 **城市间交互**：电力传输、算力迁移（可扩展为人口、资本流动）。
- 📈 **治理指标输出**：自动计算算力利用率、可再生能源占比、碳强度、电网稳定性等。
- 🧠 **知识图谱集成**：
  - 从 Neo4j 加载城市初始数据与政策。
  - 将仿真快照和交互事件写入图谱。
  - 使用 Cypher 查询政策效果，实现可追溯的仿真分析。
- ⚙️ **政策优化器**：基于随机搜索（可扩展为强化学习）自动寻找最优政策组合。

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/yourname/urban-material-actant.git
cd urban-material-actant
pip install numpy pandas neo4j geopy
```

### 运行最小示例

```python
from simulation import CityNode, CapabilityVector, StateVector, SimulationEngine, InvestmentPolicy

# 创建城市
bj = CityNode('BJ', (39.9, 116.4),
              CapabilityVector(solar_capacity=100, grid_capacity=10000, computing_power=500),
              StateVector(grid_load=8000, computing_served=400))

gz = CityNode('GZ', (26.5, 106.7),
              CapabilityVector(solar_capacity=300, grid_capacity=5000, computing_power=200),
              StateVector(grid_load=3000, computing_served=150))

# 定义政策（贵州算力补贴）
policy = InvestmentPolicy('subsidy_gz', start_time=24*30, end_time=24*365,
                          target_cities=['GZ'], target_capability='computing_efficiency',
                          increase_rate=0.2)

# 运行仿真（半年）
engine = SimulationEngine(cities=[bj, gz], policies=[policy])
results = engine.run(duration_hours=24*180)

# 查看结果
print(results.groupby('city_id')[['computing_util', 'carbon_intensity']].mean())
```

---

## 🧠 知识图谱集成（方案一：轻量级整合）

本框架支持与 Neo4j 知识图谱集成，将仿真结果持久化，并可从中加载初始数据。这是实现“仿真-知识”闭环的第一步。

### 数据模型

**节点类型**：
- `City`：城市，属性 `id`, `location`（可使用 Neo4j Point 类型）
- `Capability`：城市能力向量，与 `City` 通过 `HAS_CAPABILITY` 关系连接
- `Policy`：政策，属性 `id`, `type`, `start`, `end`, `intensity`, `target_cities`（列表）
- `Observation`：观测快照，属性 `time`, `grid_load`, `renewable_ratio`, `carbon_intensity` 等

**关系类型**：
- `(:City)-[:HAS_CAPABILITY]->(:Capability)`
- `(:City)-[:HAS_OBSERVATION]->(:Observation)`
- `(:Policy)-[:AFFECTS]->(:City)`
- `(:City)-[:INTERACTS {type, time, amount, loss}]->(:City)` （可选）

### 配置 Neo4j 客户端

```python
from simulation import Neo4jKnowledgeGraphClient, NullKnowledgeGraphClient

# 使用 Neo4j（需提前安装 neo4j 库）
kg_client = Neo4jKnowledgeGraphClient(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password"
)

# 若无 Neo4j，可使用空实现（所有写入无效果）
kg_client = NullKnowledgeGraphClient()
```

### 在仿真引擎中启用知识图谱

```python
engine = SimulationEngine(
    cities=[...],
    policies=[...],
    kg_client=kg_client,
    kg_write_frequency_hours=24  # 每天写入一次快照
)
results = engine.run(duration_hours=24*365)
```

### 查询示例

写入图谱后，可使用 Cypher 查询政策效果：

```cypher
// 查询北京在政策实施前后的碳强度变化
MATCH (c:City {id:'BJ'})-[:HAS_OBSERVATION]->(obs1:Observation {time: 720})  // 第30天
MATCH (c)-[:HAS_OBSERVATION]->(obs2:Observation {time: 1440}) // 第60天
RETURN obs1.carbon_intensity AS before, obs2.carbon_intensity AS after
```

```cypher
// 查询所有享受过补贴政策的城市及其平均可再生能源占比
MATCH (p:Policy {type:'investment'})-[:AFFECTS]->(c:City)
MATCH (c)-[:HAS_OBSERVATION]->(obs)
RETURN c.id, AVG(obs.renewable_ratio) AS avg_renewable_ratio
```

---

## 📚 文档结构

### 核心类

| 类 | 作用 |
|----|------|
| `CityNode` | 城市智能体，包含能力、状态、演化逻辑 |
| `CapabilityVector` | 物质能力（容量、效率、资源禀赋） |
| `StateVector` | 动态状态（负荷、发电、储能等） |
| `NaturalModel` | 自然条件（日照、风速、温度）生成器 |
| `InfrastructureModel` | 内部系统动力学（电网、算力负载） |
| `Policy` | 政策基类，子类实现具体干预逻辑 |
| `InteractionRule` | 城市间资源交换规则 |
| `NetworkManager` | 管理城市连接图 |
| `SimulationEngine` | 主控引擎，推进时间步 |
| `KnowledgeGraphClient` | 知识图谱客户端接口 |
| `MetricsCollector` | 收集并输出仿真指标 |

### 政策类型

| 类 | 类型 | 说明 |
|----|------|------|
| `InvestmentPolicy` | 投资型 | 增强城市特定能力，如 `computing_efficiency` |
| `RegulatoryPolicy` | 规制型 | 设置能效或排放阈值，超限惩罚 |
| `InformationPolicy` | 信息型 | 披露碳标签，影响算力需求 |
| `InfrastructurePolicy` | 基建型 | 提升网络带宽或降低输电损耗 |

### 核心治理指标

| 指标 | 含义 | 政策意义 |
|------|------|----------|
| `computing_util` | 算力利用率 | 评估算力投资效率 |
| `renewable_ratio` | 可再生能源占比 | 衡量能源清洁化，检验绿电政策 |
| `carbon_intensity` | 算力碳强度 (kg CO2/PFLOPS) | 直接关联“双碳”目标 |
| `grid_stability` | 电网稳定性 | 反映基础设施韧性 |

---

## 🔧 扩展指南

### 添加新政策类型
继承 `Policy` 类，实现 `apply` 方法，修改城市能力或状态。

```python
class NewPolicy(Policy):
    def apply(self, city, current_time):
        if self.is_active(city.id, current_time):
            # 自定义逻辑
            city.capabilities.some_attr *= (1 + self.intensity)
```

### 替换自然模型
实现 `NaturalModel` 接口，例如接入真实气象数据。

```python
class RealWeatherModel(NaturalModel):
    def get_forces(self, time_hours, location):
        # 调用外部 API 或读取历史数据
        return {'irradiance': ..., 'wind_speed': ..., 'temperature': ...}
```

### 精细化基础设施模型
实现 `InfrastructureModel` 接口，例如基于实际电网潮流的仿真。

### 新增交互规则
实现 `InteractionRule` 接口，并在 `NetworkManager` 中添加。

### 接入真实数据
实现 `DataConnector` 接口，从全国一体化算力网、国家能源局等获取数据。

### 并行加速
在 `SimulationEngine` 中，可将城市循环并行化（使用 `concurrent.futures` 或 `ray`）。

---

## 📊 示例场景：东数西算政策评估

完整示例见 `examples/east_west_policy.py`。运行后输出示例：

```
============================================================
政策效果评估报告（知识图谱版本）
============================================================

城市 BJ:
  computing_util      : 0.642 → 0.675 ↑ 5.1%
  renewable_ratio     : 0.185 → 0.210 ↑ 13.5%
  carbon_intensity    : 0.432 → 0.401 ↓ 7.2%

城市 GZ:
  computing_util      : 0.523 → 0.601 ↑ 14.9%
  renewable_ratio     : 0.623 → 0.658 ↑ 5.6%
  carbon_intensity    : 0.112 → 0.098 ↓ 12.5%
```

---

## 🤝 贡献指南

欢迎通过 Issue 和 Pull Request 贡献。主要开发方向：

- 更多政策类型（碳交易、科研资助）
- 更精细的城市内部模型（交通、人口）
- 真实数据适配器（对接公开 API）
- 可视化模块（动态地图、时序图表）
- 强化学习政策优化器

---


## 📖 引用

若本框架对你的研究有帮助，请考虑引用：

```bibtex
@software{urban_material_actant2025,
  author = {Your Name},
  title = {Urban Material Actant Network Simulation Framework},
  year = {2025},
  url = {https://github.com/yourname/urban-material-actant}
}
```

---


**让政策仿真回归物质世界——城市不仅是人的集合，更是电网、算力、阳光与风的装配。**
