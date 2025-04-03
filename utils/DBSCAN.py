import numpy as np
from DataLoader import *
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib as mpl

weighted = np.array([1, 1, 0.5])
loader = DataLoader("room1", 1)
data, frames = loader.load()

for frame in frames:
    tempData = data[data["timestamp"] == frame]
    X = weighted * tempData.iloc[:, 2:5].values
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    unique_label = np.unique(labels)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["red", "blue", "yellow", "gray", "green", "purple"]
    # 生成颜色数组，噪声点（-1）显示为灰色

    for label in unique_label:
        mask = labels == label
        if label == -1:
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c="gray", label="noise")
        else:
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                X[mask, 2],
                c=colors[label],
                label=f"cluster{label}",
            )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title(f"Timestamp: {frame}")
    plt.show()
