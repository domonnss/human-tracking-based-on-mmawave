import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.metrics.pairwise import euclidean_distances

class Track:
    """单个轨迹类，存储轨迹的状态和历史信息"""

    def __init__(self, track_id, initial_center, timestamp, max_history=30):
        self.track_id = track_id
        self.centers = [initial_center]  # 存储历史中心点 [x,y,z]
        self.timestamps = [timestamp]    # 对应的时间戳
        self.max_history = max_history   # 最大历史记录长度
        self.age = 1                     # 轨迹存在的帧数
        self.invisible_count = 0         # 连续未检测到的帧数
        self.velocity = np.zeros(3)      # 速度估计 [vx,vy,vz]
        self.predicted_center = None     # 预测的下一个中心位置
        self.color = None                # 轨迹颜色
        self.is_active = True            # 轨迹是否活跃

    def update(self, new_center, timestamp):
        """更新轨迹状态"""
        # 计算速度
        if len(self.centers) > 0:
            time_diff = (timestamp - self.timestamps[-1]).total_seconds()
            if time_diff > 0:
                self.velocity = (new_center - self.centers[-1]) / time_diff

        # 添加新的中心点和时间戳
        self.centers.append(new_center)
        self.timestamps.append(timestamp)

        # 如果超过最大历史长度，移除最旧的记录
        if len(self.centers) > self.max_history:
            self.centers.pop(0)
            self.timestamps.pop(0)

        self.age += 1
        self.invisible_count = 0
        self.is_active = True

    def predict(self, timestamp=None):
        """预测下一个位置"""
        if len(self.centers) < 2:
            self.predicted_center = self.centers[-1]
            return self.predicted_center

        if timestamp is None:
            # 使用平均时间间隔进行预测
            time_diffs = [(self.timestamps[i+1] - self.timestamps[i]).total_seconds()
                          for i in range(len(self.timestamps)-1)]
            mean_time_diff = np.mean(time_diffs) if time_diffs else 0.1
            prediction_time = mean_time_diff
        else:
            # 使用指定时间戳进行预测
            prediction_time = (timestamp - self.timestamps[-1]).total_seconds()

        # 线性预测
        self.predicted_center = self.centers[-1] + self.velocity * prediction_time
        return self.predicted_center

    def mark_invisible(self):
        """标记当前帧未检测到"""
        self.invisible_count += 1
        if self.invisible_count > 10:  # 如果连续10帧未检测到，设为非活跃
            self.is_active = False

    def get_trajectory(self):
        """获取轨迹坐标数组"""
        return np.array(self.centers)

    def get_last_center(self):
        """获取最新的中心点"""
        return self.centers[-1] if self.centers else None

    def get_first_center(self):
        """获取第一个中心点"""
        return self.centers[0] if self.centers else None

    def get_trajectory_2d(self):
        """获取轨迹在XY平面的坐标"""
        return np.array(self.centers)[:, :2] if self.centers else np.array([])


class TrackManager:
    """轨迹管理器，管理多个轨迹的创建、更新和删除"""

    def __init__(self, max_distance=1.0, min_age=3, max_invisible=10):
        self.tracks = {}          # 字典，track_id -> Track对象
        self.next_track_id = 0    # 下一个可用的轨迹ID
        self.max_distance = max_distance  # 匹配的最大距离阈值
        self.min_age = min_age    # 轨迹被认为是稳定的最小帧数
        self.max_invisible = max_invisible  # 最大连续不可见帧数
        self.colors = plt.cm.tab10.colors  # 轨迹颜色表

    def update(self, cluster_centers, labels, timestamp):
        """
        使用当前帧的聚类中心更新轨迹

        参数:
            cluster_centers: 字典，label -> center array
            labels: 当前帧中的聚类标签
            timestamp: 当前帧的时间戳
        """
        # 只处理非噪声点
        valid_labels = [label for label in labels if label != -1]

        # 如果没有活跃轨迹，则为所有簇创建新轨迹
        if not self.tracks:
            for label in valid_labels:
                center = cluster_centers[label]
                self._create_new_track(center, timestamp)
            return

        # 获取现有活跃轨迹的预测位置
        active_tracks = {tid: track for tid, track in self.tracks.items()
                         if track.is_active}

        if not active_tracks:
            # 如果没有活跃轨迹但有聚类中心，为所有中心创建新轨迹
            for label in valid_labels:
                center = cluster_centers[label]
                self._create_new_track(center, timestamp)
            return

        # 预测所有轨迹的下一个位置
        for track in active_tracks.values():
            track.predict(timestamp)

        # 计算距离矩阵（当前聚类中心与预测位置之间的距离）
        track_ids = list(active_tracks.keys())
        predictions = np.array([active_tracks[tid].predicted_center for tid in track_ids])

        if not valid_labels:
            # 如果当前帧没有有效的聚类，标记所有轨迹为不可见
            for track in active_tracks.values():
                track.mark_invisible()
            return

        # 构建用于匹配的聚类中心数组
        centers = np.array([cluster_centers[label] for label in valid_labels])

        # 计算距离矩阵
        distance_matrix = cdist(centers, predictions)

        # 使用匈牙利算法进行最优匹配
        row_indices, col_indices = linear_sum_assignment(distance_matrix)

        # 初始化标记数组，用于跟踪哪些聚类和轨迹已经匹配
        matched_clusters = set()
        matched_tracks = set()

        # 处理匹配结果
        for row, col in zip(row_indices, col_indices):
            dist = distance_matrix[row, col]
            if dist <= self.max_distance:  # 只有距离在阈值内的才算匹配成功
                cluster_label = valid_labels[row]
                track_id = track_ids[col]

                # 更新轨迹
                self.tracks[track_id].update(centers[row], timestamp)

                # 标记为已匹配
                matched_clusters.add(cluster_label)
                matched_tracks.add(track_id)

        # 为未匹配的聚类创建新轨迹
        for i, label in enumerate(valid_labels):
            if label not in matched_clusters:
                center = cluster_centers[label]
                self._create_new_track(center, timestamp)

        # 标记未匹配的轨迹为不可见
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id].mark_invisible()

        # 清理长期不可见的轨迹
        self._cleanup_tracks()

    def _create_new_track(self, center, timestamp):
        """创建新的轨迹"""
        track_id = self.next_track_id
        self.next_track_id += 1

        track = Track(track_id, center, timestamp)
        track.color = self.colors[track_id % len(self.colors)]

        self.tracks[track_id] = track
        return track_id

    def _cleanup_tracks(self):
        """清理不活跃的轨迹"""
        inactive_ids = [tid for tid, track in self.tracks.items()
                       if not track.is_active or track.invisible_count > self.max_invisible]

        for tid in inactive_ids:
            # 可以在这里添加轨迹保存逻辑，如果需要的话
            del self.tracks[tid]

    def get_active_tracks(self):
        """获取所有活跃的轨迹"""
        return {tid: track for tid, track in self.tracks.items()
                if track.is_active}

    def get_all_tracks(self):
        """获取所有轨迹"""
        return self.tracks

    def save_tracks(self, filename):
        """保存轨迹数据到文件"""
        track_data = {}
        for track_id, track in self.tracks.items():
            track_data[track_id] = {
                'centers': track.centers,
                'timestamps': track.timestamps,
                'age': track.age,
                'velocity': track.velocity,
                'is_active': track.is_active
            }

        with open(filename, 'wb') as f:
            pickle.dump(track_data, f)

    def load_tracks(self, filename):
        """从文件加载轨迹数据"""
        with open(filename, 'rb') as f:
            track_data = pickle.load(f)

        self.tracks = {}
        next_id = 0

        for track_id, data in track_data.items():
            track = Track(track_id, data['centers'][0], data['timestamps'][0])
            track.centers = data['centers']
            track.timestamps = data['timestamps']
            track.age = data['age']
            track.velocity = data['velocity']
            track.is_active = data['is_active']
            track.color = self.colors[track_id % len(self.colors)]

            self.tracks[track_id] = track
            next_id = max(next_id, track_id + 1)

        self.next_track_id = next_id

    def visualize_tracks(self, ax, show_prediction=True, show_history=True, show_velocity=True):
        """可视化轨迹"""
        active_tracks = self.get_active_tracks()

        for track_id, track in active_tracks.items():
            centers = track.get_trajectory_2d()
            if len(centers) == 0:
                continue

            color = track.color

            # 绘制历史轨迹
            if show_history and len(centers) > 1:
                ax.plot(centers[:, 0], centers[:, 1], 'o-', color=color, alpha=0.6,
                        markersize=2, linewidth=1.5, label=f'Track {track_id}')

            # 绘制当前位置
            current_pos = centers[-1]
            ax.plot(current_pos[0], current_pos[1], 'o', color=color,
                    markersize=8, label=f'Track {track_id}' if not show_history else "")

            # 显示轨迹ID
            ax.text(current_pos[0], current_pos[1] + 0.2, f"{track_id}",
                    color=color, fontsize=10, weight='bold')

            # 绘制预测位置
            if show_prediction and track.predicted_center is not None:
                pred = track.predicted_center[:2]
                ax.plot(pred[0], pred[1], 'x', color=color, markersize=8)
                ax.plot([current_pos[0], pred[0]], [current_pos[1], pred[1]],
                        '--', color=color, alpha=0.5)

            # 绘制速度
            if show_velocity and np.linalg.norm(track.velocity[:2]) > 0.01:
                vel = track.velocity[:2]
                vel_norm = np.linalg.norm(vel)
                # 绘制速度箭头
                ax.arrow(current_pos[0], current_pos[1],
                         vel[0]/vel_norm * 0.5, vel[1]/vel_norm * 0.5,
                         head_width=0.1, head_length=0.2, fc=color, ec=color, alpha=0.7)

        # 每个图只显示一次图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    def export_to_csv(self, filename):
        """将轨迹数据导出为CSV文件"""
        rows = []
        for track_id, track in self.tracks.items():
            for i, (center, timestamp) in enumerate(zip(track.centers, track.timestamps)):
                rows.append({
                    'track_id': track_id,
                    'frame': i,
                    'timestamp': timestamp,
                    'x': center[0],
                    'y': center[1],
                    'z': center[2],
                    'vx': track.velocity[0] if i > 0 else 0,
                    'vy': track.velocity[1] if i > 0 else 0,
                    'vz': track.velocity[2] if i > 0 else 0,
                    'age': track.age,
                    'is_active': track.is_active
                })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        return df

def confidence_ellipse(track, ax, n_std=2.0, **kwargs):
    """
    创建置信椭圆，用于表示轨迹的不确定性

    参数:
        track: 轨迹对象
        ax: matplotlib轴对象
        n_std: 标准差的倍数，用于确定椭圆大小
    """
    centers = track.get_trajectory_2d()
    if len(centers) < 3:
        return

    # 计算协方差矩阵
    cov = np.cov(centers[:, 0], centers[:, 1])
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    # 计算椭圆的尺寸和方向
    ell_radius_x = np.sqrt(1 + pearson) * n_std * np.sqrt(cov[0, 0])
    ell_radius_y = np.sqrt(1 - pearson) * n_std * np.sqrt(cov[1, 1])

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      **kwargs)

    # 获取坐标变换
    last_center = centers[-1]
    transform = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(1, 1) \
        .translate(last_center[0], last_center[1])

    ellipse.set_transform(transform + ax.transData)
    return ax.add_patch(ellipse)