# 城市物质行动者网络仿真框架 (Urban Material Actant Network Simulation Framework)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

一个用于**科技政策仿真**的多智能体框架，基于行动者网络理论（Actor-Network Theory, ANT），将城市抽象为封装点位，通过物质能力向量进行交互。支持投资、规制、信息、基建四类政策模拟，可接入真实数据，评估算力、能源、环境等核心治理指标。

---

## 📖 项目简介

在人工智能快速发展的背景下，科技政策的制定亟需可量化的仿真工具。本框架将**城市**视为具有物质能力的行动者（如电网容量、算力规模、可再生能源禀赋），通过多智能体建模模拟城市间资源流动与政策干预效果。核心创新在于：

- **ANT 广义对称性**：城市、自然条件、基础设施均为平等行动者，通过状态变化交互。
- **有限复杂抽象**：以城市为最小分析单元，内部用系统动力学聚合，避免微观细节爆炸。
- **政策可计算化**：将政策转化为可参数化、可组合、可动态调整的代码模块。
- **治理指标对齐**：直接输出算力利用率、碳强度、可再生能源占比、电网稳定性等决策关键指标。

---

## ✨ 特性

- 🏙️ **城市点位建模**：每个城市封装为 `CityNode`，包含物质能力（`CapabilityVector`）和动态状态（`StateVector`）。
- 🔌 **可插拔模型**：自然条件、基础设施演化、城市交互规则均基于抽象接口，可自由替换。
- 📊 **四类政策模拟**：
  - **投资型**：补贴、税收优惠，增强特定能力。
  - **规制型**：能效标准、碳排放限额，附带惩罚机制。
  - **信息型**：碳标签、能效标识，影响需求侧行为。
  - **基建型**：电网互联、网络扩容，降低传输损耗。
- ⏱️ **政策执行细节**：支持执行差距、时间滞后、动态自适应强度。
- 🔗 **城市间交互**：电力传输、算力迁移（可扩展为人口、资本流动）。
- 📈 **治理指标输出**：自动计算算力利用率、可再生能源占比、碳强度、电网稳定性等。
- 🌐 **外部数据接入**：通过 `DataConnector` 对接真实数据（算力监测平台、能源局报告等）。
- ⚙️ **政策优化器**：基于随机搜索（可扩展为强化学习）自动寻找最优政策组合。

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/yourname/urban-material-actant.git
cd urban-material-actant
pip install -r requirements.txt
```

依赖项：
- Python 3.8+
- numpy
- pandas
- geopy
- matplotlib (可选，用于可视化)

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

# 运行仿真
engine = SimulationEngine(cities=[bj, gz], policies=[policy])
results = engine.run(duration_hours=24*180)  # 半年

# 查看结果
print(results.groupby('city_id')[['computing_util', 'carbon_intensity']].mean())
```

---

## 📚 文档

### 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Simulation Engine                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Clock  │  │ Policy  │  │Network  │  │Metrics  │        │
│  │ (tick)  │  │Manager  │  │Manager  │  │Collector│        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       └────────────┼─────────────┼────────────┘              │
│                    ▼             ▼                            │
│           ┌────────────────────────────┐                     │
│           │      CityNode 列表         │                     │
│           │  [City_1, City_2, ...]    │                     │
│           └────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 主要组件

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
| `DataConnector` | 外部数据接入接口 |
| `PolicyOptimizer` | 政策参数自动搜索 |

### 扩展指南

1. **添加新政策类型**：继承 `Policy` 类，实现 `apply` 方法，修改城市能力或状态。
2. **替换自然模型**：实现 `NaturalModel` 接口，例如接入气象数据API。
3. **精细化基础设施模型**：实现 `InfrastructureModel`，例如基于实际电网潮流的仿真。
4. **新增交互规则**：实现 `InteractionRule`，添加至 `NetworkManager`。
5. **接入真实数据**：实现 `DataConnector`，从全国一体化算力网、国家能源局等获取数据。

---

## 📊 治理观测指标

仿真自动输出以下核心指标（在 `CityNode.get_metrics()` 中定义）：

| 指标 | 含义 | 政策意义 |
|------|------|----------|
| `computing_util` | 算力利用率 = 已服务算力 / 总算力 | 评估算力投资效率，引导需求 |
| `renewable_ratio` | 可再生能源占比 = 可再生发电 / 总负荷 | 衡量能源清洁化，检验绿电政策 |
| `carbon_intensity` | 算力碳强度 = (火电 * 排放因子) / 算力 | 直接关联“双碳”目标 |
| `grid_stability` | 电网稳定性 = 1 - 负荷偏离50%容量的程度 | 反映基础设施韧性 |
| `battery_soc` | 储能荷电状态 | 储能调节能力 |

---

## 🧪 示例场景

### 东数西算政策评估

模拟“东数西算”工程中，对西部算力中心补贴、东部能效标准、东西部网络扩容的组合政策效果。

```python
# 完整示例见 examples/east_west_policy.py
```

输出报告示例：
```
============================================================
政策效果评估报告
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

### 政策参数优化

通过随机搜索寻找最优补贴强度和启动时间。

```python
optimizer = PolicyOptimizer(engine, policy_space={...}, objective_func=lambda df: df['renewable_ratio'].mean())
best = optimizer.random_search(n_trials=50)
```

---

## 🤝 贡献指南

欢迎通过 Issue 和 Pull Request 贡献代码。主要开发方向：

- 更多政策类型（如碳交易、科研资助）
- 更精细的城市内部模型（如交通、人口）
- 真实数据适配器（对接公开API）
- 可视化模块（动态地图、时序图表）
- 并行加速（多进程/Ray）

---

## 📖 引用

若本框架对你的研究有帮助，请考虑引用：

```
@software{urban_material_actant2025,
  author = {Your Name},
  title = {Urban Material Actant Network Simulation Framework},
  year = {2025},
  url = {https://github.com/yourname/urban-material-actant}
}
```



---

**让政策仿真回归物质世界——城市不仅是人的集合，更是电网、算力、阳光与风的装配。**
