import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from utils.DBSCAN import run_dbscan_tracking, DEFAULT_MIN_FRAMES

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行DBSCAN聚类和轨迹跟踪')
    parser.add_argument('--eps', type=float, default=0.3, help='DBSCAN的邻域半径')
    parser.add_argument('--min_samples', type=int, default=5, help='DBSCAN的最小样本数')
    parser.add_argument('--room', type=str, default='room1', help='房间名称')
    parser.add_argument('--count', type=int, default=1, help='人数')
    parser.add_argument('--max_distance', type=float, default=1.0, help='最大匹配距离')
    parser.add_argument('--max_invisible', type=int, default=10, help='最大不可见帧数')
    parser.add_argument('--save_animation', action='store_true', help='是否保存动画')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--min_frames', type=int, default=DEFAULT_MIN_FRAMES, help='最小帧数阈值')

    args = parser.parse_args()

    # 运行DBSCAN聚类和轨迹跟踪
    run_dbscan_tracking(
        eps=args.eps,
        min_samples=args.min_samples,
        room=args.room,
        count_person=args.count,
        max_distance=args.max_distance,
        max_invisible=args.max_invisible,
        save_animation=args.save_animation,
        output_dir=args.output_dir,
        min_frames=args.min_frames
    )

if __name__ == "__main__":
    main()