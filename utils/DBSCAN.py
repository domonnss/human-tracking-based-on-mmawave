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

# default parameters
DEFAULT_WEIGHTED = np.array([1, 1, 0.5])   # x,y,z三轴权重
DEFAULT_EPS = 0.3                          # DBSCAN 半径
DEFAULT_MIN_SAMPLES = 10                    # DBSCAN 最小样本数
DEFAULT_ROOM = "room1"                     # 默认房间
DEFAULT_COUNT_PERSON = 1                   # 默认人数
DEFAULT_MAX_DISTANCE = 1.0                 # 最大匹配距离
DEFAULT_MAX_INVISIBLE = 20                 # 最大不可见帧数
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
    ax1.set_title("Real-time Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # 2D视图 - 显示轨迹
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("trajectory")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True, linestyle='--', alpha=0.7)

    frames_for_animation = []


    # Color list
    colors = ["red", "blue", "yellow", "green", "purple", "orange", "cyan", "magenta", "brown", "pink"]

    print(f"Process {len(frames)} frames of data...")

    for frame_idx, frame in enumerate(frames):
        ax1.cla()

        tempData = data[data["timestamp"] == frame]

        if (tempData["# Obj"] < min_samples).any():
            continue

        X = weighted * tempData.iloc[:, 2:5].values
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        unique_label = np.unique(labels)

        current_frame_centers = {}

        for label in unique_label:
            if label == -1:
                continue

            mask = labels == label
            point = X[mask, :]

            center = np.mean(point, axis=0)
            current_frame_centers[label] = center[:2]

            # plot point cloud
            ax1.scatter(
                point[:, 0], point[:, 1], point[:, 2],
                c=colors[label % len(colors)],
                label=f"cluster {label}",
                alpha=0.8
            )

            x_min, x_max = point[:, 0].min(), point[:, 0].max()
            y_min, y_max = point[:, 1].min(), point[:, 1].max()
            z_min, z_max = point[:, 2].min(), point[:, 2].max()

            # draw boundry
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

        noise_mask = labels == -1
        if np.any(noise_mask):
            noise_points = X[noise_mask, :]
            ax1.scatter(
                noise_points[:, 0], noise_points[:, 1], noise_points[:, 2],
                c="gray", label="noise", alpha=0.3, marker="."
            )

        track_manager.update(current_frame_centers, unique_label, frame)

        ax2.cla()
        ax2.set_title("人员轨迹")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.grid(True, linestyle='--', alpha=0.7)

        track_manager.visualize_tracks(ax2, show_prediction=True, show_history=True, show_velocity=True)

        # confidence ellipse
        for track_id, track in track_manager.get_active_tracks().items():
            if track.age > min_frames:
                confidence_ellipse(
                    track, ax2, n_std=3.0,
                    facecolor=track.color, alpha=0.2, edgecolor=track.color
                )

        ax2.set_xlim(-5, 5)
        ax2.set_ylim(0, 10)

        fig.suptitle(f"room: {room}, person count: {count_person}, frame: {frame_idx+1}/{len(frames)}, TimeStamp: {frame}")

        # 更新图形
        ax1.set_title(f"{len(unique_label)} point cloud clustering")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend(loc="upper right")

        plt.draw()
        plt.pause(0.01)

        # cunrrent frame capture
        if save_animation:
            frames_for_animation.append(plt.gcf())

    # save tracks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    track_manager.save_tracks(os.path.join(output_dir, f"tracks_{timestamp}.csv"))

    if save_animation and frames_for_animation:
        print("Saving animation...")
        ani = animation.ArtistAnimation(fig, frames_for_animation, interval=100, blit=True)
        ani.save(os.path.join(output_dir, f"animation_{timestamp}.mp4"), writer='ffmpeg')
        print("Animation saved")

    plt.ioff()
    plt.show()
