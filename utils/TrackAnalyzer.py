import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import os
import argparse
from datetime import datetime, timedelta
from scipy import stats

class TrackAnalyzer:
    """轨迹分析工具，用于可视化和分析保存的轨迹数据"""

    def __init__(self, track_file=None, csv_file=None):
        """
        初始化分析器

        参数:
            track_file: 轨迹pkl文件路径
            csv_file: 轨迹CSV文件路径
        """
        self.track_data = None
        self.df = None

        if track_file and os.path.exists(track_file):
            self.load_track_data(track_file)

        if csv_file and os.path.exists(csv_file):
            self.load_csv_data(csv_file)

    def load_track_data(self, track_file):
        """加载轨迹pkl数据"""
        with open(track_file, 'rb') as f:
            self.track_data = pickle.load(f)
        print(f"加载了 {len(self.track_data)} 条轨迹")
        return self.track_data

    def load_csv_data(self, csv_file):
        """加载轨迹CSV数据"""
        self.df = pd.read_csv(csv_file)
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        print(f"加载了CSV数据，共 {len(self.df)} 行")
        return self.df

    def plot_all_trajectories(self, show_labels=True, ax=None, title="所有轨迹", grid=True):
        """绘制所有轨迹"""
        if self.df is None:
            print("请先加载CSV数据")
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # 获取所有唯一轨迹ID
        track_ids = self.df['track_id'].unique()

        # 为每个轨迹ID分配一个颜色
        colors = plt.cm.tab10.colors

        # 绘制每条轨迹
        for i, track_id in enumerate(track_ids):
            track_data = self.df[self.df['track_id'] == track_id]
            ax.plot(track_data['x'], track_data['y'], 'o-',
                    color=colors[i % len(colors)],
                    alpha=0.7, linewidth=2, markersize=4,
                    label=f"轨迹 {track_id}")

            # 标记起点和终点
            ax.plot(track_data['x'].iloc[0], track_data['y'].iloc[0], 'o',
                    color=colors[i % len(colors)], markersize=8)
            ax.plot(track_data['x'].iloc[-1], track_data['y'].iloc[-1], 's',
                    color=colors[i % len(colors)], markersize=8)

            # 添加轨迹ID标签
            if show_labels:
                mid_point = len(track_data) // 2
                ax.text(track_data['x'].iloc[mid_point],
                        track_data['y'].iloc[mid_point],
                        str(track_id), fontsize=12,
                        color=colors[i % len(colors)],
                        fontweight='bold')

        ax.set_title(title)
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")

        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)

        # 调整轴的范围，添加一些边距
        x_min, x_max = self.df['x'].min(), self.df['x'].max()
        y_min, y_max = self.df['y'].min(), self.df['y'].max()
        margin = 0.5
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        # 添加图例
        if len(track_ids) < 10:  # 只有当轨迹数量合理时才显示图例
            ax.legend(loc='best')

        return ax

    def animate_trajectories(self, output_file=None, interval=100, fps=10):
        """创建轨迹动画"""
        if self.df is None:
            print("请先加载CSV数据")
            return

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))

        # 获取所有唯一轨迹ID和时间戳
        track_ids = self.df['track_id'].unique()
        timestamps = sorted(self.df['timestamp'].unique())

        # 为每个轨迹ID分配一个颜色
        colors = plt.cm.tab10.colors

        # 轨迹历史数据
        trajectory_history = {track_id: [] for track_id in track_ids}

        # 更新函数
        def update(frame):
            ax.clear()
            current_time = timestamps[frame]

            # 在标题中显示当前时间
            ax.set_title(f"轨迹动画 - 时间: {current_time}")

            # 获取当前时间戳之前的所有数据
            current_data = self.df[self.df['timestamp'] <= current_time]

            # 更新每条轨迹的历史
            for track_id in track_ids:
                track_data = current_data[current_data['track_id'] == track_id]
                if len(track_data) > 0:
                    # 保存轨迹点
                    trajectory_history[track_id] = list(zip(track_data['x'], track_data['y']))

                    # 只绘制有数据的轨迹
                    if trajectory_history[track_id]:
                        # 转换为numpy数组以便绘图
                        trajectory = np.array(trajectory_history[track_id])

                        # 绘制轨迹线
                        ax.plot(trajectory[:, 0], trajectory[:, 1], '-',
                                color=colors[track_id % len(colors)],
                                alpha=0.7, linewidth=2)

                        # 绘制轨迹点
                        ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                                color=colors[track_id % len(colors)],
                                s=50, label=f"轨迹 {track_id}")

                        # 显示轨迹ID
                        ax.text(trajectory[-1, 0], trajectory[-1, 1] + 0.2,
                                str(track_id), fontsize=10,
                                color=colors[track_id % len(colors)])

            ax.set_xlabel("X 坐标")
            ax.set_ylabel("Y 坐标")
            ax.grid(True, linestyle='--', alpha=0.7)

            # 设置轴的范围
            x_min, x_max = self.df['x'].min(), self.df['x'].max()
            y_min, y_max = self.df['y'].min(), self.df['y'].max()
            margin = 0.5
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)

            return ax,

        # 创建动画
        ani = FuncAnimation(fig, update, frames=len(timestamps),
                            interval=interval, blit=True)

        # 保存或显示动画
        if output_file:
            ani.save(output_file, writer='ffmpeg', fps=fps)
            print(f"动画已保存至: {output_file}")
        else:
            plt.show()

        return ani

    def calculate_statistics(self):
        """计算轨迹统计信息"""
        if self.df is None:
            print("请先加载CSV数据")
            return None

        stats_dict = {}
        track_ids = self.df['track_id'].unique()

        for track_id in track_ids:
            track_data = self.df[self.df['track_id'] == track_id]

            # 计算轨迹长度（路径总长度）
            points = np.array(list(zip(track_data['x'], track_data['y'])))
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            total_distance = np.sum(distances)

            # 计算持续时间
            if 'timestamp' in track_data.columns:
                duration = (track_data['timestamp'].max() - track_data['timestamp'].min()).total_seconds()
            else:
                duration = len(track_data) - 1  # 假设每帧之间的时间间隔为1

            # 计算平均速度
            avg_speed = total_distance / max(duration, 1e-6)

            # 计算最大速度
            if 'vx' in track_data.columns and 'vy' in track_data.columns:
                speeds = np.sqrt(track_data['vx']**2 + track_data['vy']**2)
                max_speed = speeds.max()
            else:
                max_speed = np.nan

            # 直线距离（起点到终点）
            start_point = points[0]
            end_point = points[-1]
            straight_distance = np.sqrt(np.sum((end_point - start_point)**2))

            # 路径效率（直线距离/实际路径长度）
            path_efficiency = straight_distance / max(total_distance, 1e-6)

            stats_dict[track_id] = {
                '轨迹ID': track_id,
                '点数': len(track_data),
                '路径长度': total_distance,
                '持续时间(秒)': duration,
                '平均速度': avg_speed,
                '最大速度': max_speed,
                '直线距离': straight_distance,
                '路径效率': path_efficiency
            }

        # 转换为DataFrame
        stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')
        return stats_df

    def plot_speed_profile(self, track_id=None):
        """绘制速度剖面图"""
        if self.df is None:
            print("请先加载CSV数据")
            return

        if 'vx' not in self.df.columns or 'vy' not in self.df.columns:
            print("CSV数据中没有速度信息")
            return

        # 计算速度大小
        self.df['speed'] = np.sqrt(self.df['vx']**2 + self.df['vy']**2)

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))

        if track_id is not None:
            # 绘制特定轨迹的速度剖面
            track_data = self.df[self.df['track_id'] == track_id]

            if len(track_data) == 0:
                print(f"未找到ID为 {track_id} 的轨迹")
                return

            # 时间序列
            if 'timestamp' in track_data.columns:
                x = (track_data['timestamp'] - track_data['timestamp'].iloc[0]).dt.total_seconds()
                xlabel = "时间 (秒)"
            else:
                x = track_data['frame']
                xlabel = "帧序号"

            ax.plot(x, track_data['speed'], 'o-', linewidth=2, markersize=4, label=f"轨迹 {track_id}")
            ax.set_title(f"轨迹 {track_id} 的速度剖面")

        else:
            # 绘制所有轨迹的速度剖面
            track_ids = self.df['track_id'].unique()
            colors = plt.cm.tab10.colors

            for i, tid in enumerate(track_ids):
                track_data = self.df[self.df['track_id'] == tid]

                # 时间序列
                if 'timestamp' in track_data.columns:
                    x = (track_data['timestamp'] - track_data['timestamp'].iloc[0]).dt.total_seconds()
                    xlabel = "时间 (秒)"
                else:
                    x = track_data['frame']
                    xlabel = "帧序号"

                ax.plot(x, track_data['speed'], 'o-', color=colors[i % len(colors)],
                        linewidth=2, markersize=4, label=f"轨迹 {tid}")

            ax.set_title("所有轨迹的速度剖面")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("速度")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')

        return fig, ax

    def plot_heatmap(self, resolution=0.1):
        """绘制轨迹热图"""
        if self.df is None:
            print("请先加载CSV数据")
            return

        # 创建网格
        x_min, x_max = self.df['x'].min(), self.df['x'].max()
        y_min, y_max = self.df['y'].min(), self.df['y'].max()

        x_bins = np.arange(x_min, x_max + resolution, resolution)
        y_bins = np.arange(y_min, y_max + resolution, resolution)

        # 使用numpy的histogram2d函数创建热图数据
        heatmap, xedges, yedges = np.histogram2d(
            self.df['x'], self.df['y'], bins=[x_bins, y_bins]
        )

        # 转置热图，使其符合图像坐标系
        heatmap = heatmap.T

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制热图
        c = ax.pcolormesh(xedges, yedges, heatmap, cmap='hot', alpha=0.7)
        fig.colorbar(c, ax=ax, label='点密度')

        # 在热图上绘制轨迹线
        self.plot_all_trajectories(show_labels=False, ax=ax, title="轨迹热图", grid=False)

        return fig, ax

    def export_statistics(self, output_file):
        """导出统计信息到CSV文件"""
        stats_df = self.calculate_statistics()
        if stats_df is not None:
            stats_df.to_csv(output_file)
            print(f"统计信息已导出至: {output_file}")
            return stats_df
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="轨迹分析工具")
    parser.add_argument("--track", type=str, help="轨迹PKL文件路径")
    parser.add_argument("--csv", type=str, help="轨迹CSV文件路径")
    parser.add_argument("--animate", action="store_true", help="创建轨迹动画")
    parser.add_argument("--output", type=str, default="analysis_results", help="输出目录")
    parser.add_argument("--track_id", type=int, help="指定分析的轨迹ID")
    parser.add_argument("--heatmap", action="store_true", help="生成热图")
    parser.add_argument("--speed", action="store_true", help="生成速度剖面图")
    parser.add_argument("--stats", action="store_true", help="计算统计信息")

    args = parser.parse_args()

    # 确保至少有一个输入文件
    if not args.track and not args.csv:
        print("请指定至少一个轨迹文件 (--track 或 --csv)")
        return

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 创建分析器
    analyzer = TrackAnalyzer(track_file=args.track, csv_file=args.csv)

    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 默认行为：如果没有指定其他选项，则生成所有图表
    if not any([args.animate, args.heatmap, args.speed, args.stats]):
        args.animate = args.heatmap = args.speed = args.stats = True

    # 绘制轨迹图
    plt.figure(figsize=(10, 8))
    analyzer.plot_all_trajectories()
    plt.savefig(os.path.join(args.output, f"trajectories_{timestamp}.png"), dpi=300)
    print(f"轨迹图已保存至: {os.path.join(args.output, f'trajectories_{timestamp}.png')}")

    # 创建动画
    if args.animate:
        animation_file = os.path.join(args.output, f"animation_{timestamp}.mp4")
        analyzer.animate_trajectories(output_file=animation_file)

    # 生成热图
    if args.heatmap:
        fig, _ = analyzer.plot_heatmap()
        heatmap_file = os.path.join(args.output, f"heatmap_{timestamp}.png")
        fig.savefig(heatmap_file, dpi=300)
        print(f"热图已保存至: {heatmap_file}")

    # 生成速度剖面图
    if args.speed:
        fig, _ = analyzer.plot_speed_profile(args.track_id)
        speed_file = os.path.join(args.output, f"speed_profile_{timestamp}.png")
        fig.savefig(speed_file, dpi=300)
        print(f"速度剖面图已保存至: {speed_file}")

    # 计算统计信息
    if args.stats:
        stats_file = os.path.join(args.output, f"statistics_{timestamp}.csv")
        analyzer.export_statistics(stats_file)

    # 显示所有图表
    plt.show()

if __name__ == "__main__":
    main()