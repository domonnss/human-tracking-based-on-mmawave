import numpy as np
from DataLoader import *
from GetClusterCenter import *
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

weighted = np.array([1, 1, 0.5])
loader = DataLoader("room1", 1)
data, frames = loader.load()

plt.ion()
fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_title("3D Plot")

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("2D Plot")

# init
colors = ["red", "blue", "yellow", "gray", "green", "purple"]
cluster_centers = []

for frame in frames:
    ax1.cla()
    tempData = data[data["timestamp"] == frame]
    if (tempData["# Obj"] < 5).any():
        continue
    X = weighted * tempData.iloc[:, 2:5].values
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    unique_label = np.unique(labels)

    # 绘制当前帧
    for label in unique_label:
        if len(unique_label) != 2:
            continue
        mask = labels == label
        point = X[mask, :]
        if label == -1:
            ax1.scatter(point[:, 0], point[:, 1], point[:, 2], c="gray", label="noise")
        else:
            ax1.scatter(
                point[:, 0],
                point[:, 1],
                point[:, 2],
                c=colors[label % len(colors)],  # 循环使用颜色
                label=f"cluster{label}",
            )
            # 绘制包围盒
            x_min, x_max = point[:, 0].min(), point[:, 0].max()
            y_min, y_max = point[:, 1].min(), point[:, 1].max()
            z_min, z_max = point[:, 2].min(), point[:, 2].max()
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
                    edge[0],
                    edge[1],
                    edge[2],
                    linestyle="--",
                    alpha=0.5,
                    c=colors[label % len(colors)],
                )
            center = get_cluster_centers(cluster_centers, point)
            ax2.plot(
                center[:, 0],
                center[:, 1],
                marker="x",
                c=colors[label % len(colors)],
                label=f"cluster {label}",
            )

    # ax1
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend(loc="upper right")
    ax1.set_title(f"Timestamp: {frame}")

    # ax2
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("trajectory")

    plt.draw()
    plt.pause(0.5)

plt.ioff()
plt.show()

# if __name__ == "__main__":
