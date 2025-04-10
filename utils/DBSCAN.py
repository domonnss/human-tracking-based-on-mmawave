import numpy as np
from utils.DataLoader import DataLoader
from utils.GetClusterCenter import get_cluster_centers
from utils.TrackManagement import TrackManager, confidence_ellipse
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse
from datetime import datetime

# 默认参数
DEFAULT_WEIGHTED = np.array([1, 1, 0.5])  # 坐标权重
DEFAULT_EPS = 0.5                          # DBSCAN 半径
DEFAULT_MIN_SAMPLES = 5                    # DBSCAN 最小样本数
DEFAULT_ROOM = "room1"                     # 默认房间
DEFAULT_COUNT_PERSON = 1                   # 默认人数
DEFAULT_MAX_DISTANCE = 1.0                 # 最大匹配距离
DEFAULT_MAX_INVISIBLE = 10                 # 最大不可见帧数
DEFAULT_SAVE_ANIMATION = False             # 是否保存动画
DEFAULT_MIN_FRAMES = 3                     # 最小帧数阈值

def run_dbscan_tracking(
    weighted=DEFAULT_WEIGHTED,
    eps=DEFAULT_EPS,
    min_samples=DEFAULT_MIN_SAMPLES,
    room=DEFAULT_ROOM,
    count_person=DEFAULT_COUNT_PERSON,
    max_distance=DEFAULT_MAX_DISTANCE,
    max_invisible=DEFAULT_MAX_INVISIBLE,
    save_animation=DEFAULT_SAVE_ANIMATION,
    output_dir="results",
    min_frames=DEFAULT_MIN_FRAMES
):
    """运行DBSCAN聚类和轨迹管理"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    loader = DataLoader(room, count_person)
    data, frames = loader.load()

    # 初始化轨迹管理器
    track_manager = TrackManager(
        max_distance=max_distance,
        max_invisible=max_invisible
    )

    # 初始化图形
    plt.ion()
    fig = plt.figure(figsize=(15, 7))

    # 3D视图 - 显示当前聚类点云
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.set_title("DBSCAN聚类 - 实时点云")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # 2D视图 - 显示轨迹
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("人员轨迹")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 保存所有帧用于制作动画
    frames_for_animation = []

    # 用于保存所有聚类中心的字典
    cluster_centers_dict = {}

    # 颜色列表
    colors = ["red", "blue", "yellow", "green", "purple", "orange", "cyan", "magenta", "brown", "pink"]

    print(f"处理 {len(frames)} 帧数据...")

    # 处理每一帧
    for frame_idx, frame in enumerate(frames):
        # 清除3D图形
        ax1.cla()

        # 获取当前帧的数据
        tempData = data[data["timestamp"] == frame]

        # 如果对象数量不足，跳过此帧
        if (tempData["# Obj"] < min_samples).any():
            continue

        # 应用权重并执行DBSCAN聚类
        X = weighted * tempData.iloc[:, 2:5].values
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        unique_label = np.unique(labels)

        # 计算每个聚类的中心点
        current_frame_centers = {}

        for label in unique_label:
            if label == -1:  # 跳过噪声点
                continue

            mask = labels == label
            point = X[mask, :]

            # 计算聚类中心
            center = np.mean(point, axis=0)
            current_frame_centers[label] = center[:2]  # 只使用x,y坐标

            # 绘制3D点云和聚类边界框
            ax1.scatter(
                point[:, 0], point[:, 1], point[:, 2],
                c=colors[label % len(colors)],
                label=f"cluster {label}",
                alpha=0.8
            )

            # 计算边界
            x_min, x_max = point[:, 0].min(), point[:, 0].max()
            y_min, y_max = point[:, 1].min(), point[:, 1].max()
            z_min, z_max = point[:, 2].min(), point[:, 2].max()

            # 绘制边界框
            edges = [
                # 底面
                ((x_min, x_max), (y_min, y_min), (z_min, z_min)),
                ((x_min, x_max), (y_max, y_max), (z_min, z_min)),
                ((x_min, x_min), (y_min, y_max), (z_min, z_min)),
                ((x_max, x_max), (y_min, y_max), (z_min, z_min)),
                # 顶面
                ((x_min, x_max), (y_min, y_min), (z_max, z_max)),
                ((x_min, x_max), (y_max, y_max), (z_max, z_max)),
                ((x_min, x_min), (y_min, y_max), (z_max, z_max)),
                ((x_max, x_max), (y_min, y_max), (z_max, z_max)),
                # 侧面
                ((x_min, x_min), (y_min, y_min), (z_min, z_max)),
                ((x_max, x_max), (y_min, y_min), (z_min, z_max)),
                ((x_min, x_min), (y_max, y_max), (z_min, z_max)),
                ((x_max, x_max), (y_max, y_max), (z_min, z_max)),
            ]

            for edge in edges:
                ax1.plot3D(
                    edge[0], edge[1], edge[2],
                    linestyle="--", alpha=0.5,
                    c=colors[label % len(colors)]
                )

        # 绘制噪声点
        noise_mask = labels == -1
        if np.any(noise_mask):
            noise_points = X[noise_mask, :]
            ax1.scatter(
                noise_points[:, 0], noise_points[:, 1], noise_points[:, 2],
                c="gray", label="noise", alpha=0.3, marker="."
            )

        # 更新轨迹管理器
        track_manager.update(current_frame_centers, unique_label, frame)

        # 清除轨迹图
        ax2.cla()
        ax2.set_title("人员轨迹")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 可视化轨迹
        track_manager.visualize_tracks(ax2, show_prediction=True, show_history=True, show_velocity=True)

        # 添加置信椭圆
        for track_id, track in track_manager.get_active_tracks().items():
            if track.age > min_frames:  # 使用参数控制最小帧数
                confidence_ellipse(
                    track, ax2, n_std=3.0,
                    facecolor=track.color, alpha=0.2, edgecolor=track.color
                )

        # 设置轴的范围
        ax2.set_xlim(-5, 5)
        ax2.set_ylim(0, 10)

        # 添加标题
        fig.suptitle(f"房间: {room}, 人数: {count_person}, 帧: {frame_idx+1}/{len(frames)}, 时间戳: {frame}")

        # 更新图形
        ax1.set_title(f"聚类点云 (共{len(unique_label)}个聚类)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend(loc="upper right")

        # 显示进度
        if frame_idx % 10 == 0:
            print(f"已处理 {frame_idx}/{len(frames)} 帧 ({frame_idx/len(frames)*100:.1f}%)")

        # 绘制并暂停
        plt.draw()
        plt.pause(0.01)

        # 如果保存动画，需要捕获当前帧
        if save_animation:
            frames_for_animation.append(plt.gcf())

    # 保存轨迹数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    track_manager.save_tracks(os.path.join(output_dir, f"tracks_{timestamp}.csv"))

    # 如果保存动画，创建动画文件
    if save_animation and frames_for_animation:
        print("正在保存动画...")
        ani = animation.ArtistAnimation(fig, frames_for_animation, interval=100, blit=True)
        ani.save(os.path.join(output_dir, f"animation_{timestamp}.mp4"), writer='ffmpeg')
        print("动画保存完成")

    # 保持图形显示
    plt.ioff()
    plt.show()
