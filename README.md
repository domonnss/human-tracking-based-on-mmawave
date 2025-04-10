# 行人轨迹管理与分析系统

这个项目实现了一个基于毫米波雷达数据的行人轨迹管理与分析系统。系统使用DBSCAN聚类算法识别数据中的人员位置，并通过跨帧匹配算法跟踪人员的移动轨迹。

## 功能特点

- 基于DBSCAN的人员位置聚类
- 跨帧的轨迹管理与追踪
- 轨迹ID的持久化和管理
- 速度和方向的估计
- 轨迹可视化和分析
- 轨迹统计数据导出

## 目录结构

```
├── utils/                      # 核心功能模块
│   ├── DBSCAN.py               # DBSCAN聚类和轨迹处理主程序
│   ├── DataLoader.py           # 数据加载模块
│   ├── GetClusterCenter.py     # 聚类中心计算
│   ├── TrackManagement.py      # 轨迹管理模块
│   ├── TrackAnalyzer.py        # 轨迹分析工具
│   └── WeightedNorm.py         # 加权距离计算
├── people-gait/                # 数据目录
│   ├── room1/                  # 房间1数据
│   └── room2/                  # 房间2数据
├── utils_notebook/             # Jupyter实验脚本
└── results/                    # 结果输出目录
```

## 使用方法

### 运行轨迹跟踪

基本用法:

```bash
python utils/DBSCAN.py
```

参数选项:

```bash
python utils/DBSCAN.py --eps 0.5 --min_samples 5 --room room1 --count 1 --save_animation
```

参数说明:
- `--eps`: DBSCAN半径参数 (默认 0.5)
- `--min_samples`: DBSCAN最小样本数 (默认 5)
- `--room`: 房间编号 (默认 "room1")
- `--count`: 人数 (默认 1)
- `--max_distance`: 轨迹匹配最大距离 (默认 1.0)
- `--max_invisible`: 最大不可见帧数 (默认 10)
- `--save_animation`: 保存动画 (可选)
- `--output`: 输出目录 (默认 "results")

### 分析轨迹数据

运行轨迹分析器:

```bash
python utils/TrackAnalyzer.py --csv results/tracks_room1_1_20230415_123045.csv
```

分析选项:
- `--track`: 轨迹PKL文件路径
- `--csv`: 轨迹CSV文件路径
- `--animate`: 创建轨迹动画
- `--heatmap`: 生成热图
- `--speed`: 生成速度剖面图
- `--stats`: 计算统计信息
- `--track_id`: 指定分析的轨迹ID
- `--output`: 输出目录 (默认 "analysis_results")

## 轨迹管理技术细节

系统使用以下算法对轨迹进行管理:

1. **DBSCAN聚类**: 使用DBSCAN算法将雷达点云数据聚类，识别人员位置
2. **跨帧匹配**: 使用匈牙利算法(线性分配问题)将当前帧的聚类中心与预测位置匹配
3. **轨迹预测**: 使用线性速度模型预测下一帧的位置
4. **轨迹平滑**: 使用历史数据平滑轨迹，减少噪声影响

## 轨迹分析功能

- **轨迹可视化**: 2D和3D轨迹可视化
- **轨迹动画**: 生成随时间变化的轨迹动画
- **热图分析**: 创建轨迹密度热图
- **速度剖面**: 分析行人移动速度变化
- **统计分析**: 计算轨迹长度、速度、效率等指标

## 依赖项

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- SciPy
- Seaborn

安装依赖:

```bash
pip install numpy pandas matplotlib scikit-learn scipy seaborn
```

## 示例

1. 运行跟踪:

```bash
python utils/DBSCAN.py --room room1 --count 1 --eps 0.5 --min_samples 5
```

2. 分析轨迹:

```bash
python utils/TrackAnalyzer.py --csv results/tracks_room1_1_*.csv --animate --heatmap --speed --stats
```

## 注意事项

- 确保`ffmpeg`已安装，以支持动画生成
- 数据应遵循预期的格式，以确保程序正常运行
- 调整DBSCAN参数可能需要根据实际数据特性进行微调