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

    def __init__(self, track_file=None, csv_file=None):

        self.track_data = None
        self.df = None

        if track_file and os.path.exists(track_file):
            self.load_track_data(track_file)

        if csv_file and os.path.exists(csv_file):
            self.load_csv_data(csv_file)

    def load_track_data(self, track_file):
        with open(track_file, 'rb') as f:
            self.track_data = pickle.load(f)
        print(f"load {len(self.track_data)} tracks")
        return self.track_data

    def load_csv_data(self, csv_file):
        self.df = pd.read_csv(csv_file)
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        return self.df

    def plot_all_trajectories(self, show_labels=True, ax=None, title="alltrajectories"):
        if self.df is None:
            print("load CSV file first")
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # get unique track ID
        # stored in col "track id"
        track_ids = self.df['track_id'].unique()

        colors = plt.cm.tab10.colors

        # plot trajectories
        for i, track_id in enumerate(track_ids):
            track_data = self.df[self.df['track_id'] == track_id]
            ax.plot(track_data['x'], track_data['y'], 'o-',
                    color=colors[i % len(colors)],
                    alpha=0.7, linewidth=2, markersize=1,
                    label=f"track {track_id}")

            # Mark start and end points
            ax.plot(track_data['x'].iloc[0], track_data['y'].iloc[0], 'o',
                    color=colors[i % len(colors)], markersize=8)
            ax.plot(track_data['x'].iloc[-1], track_data['y'].iloc[-1], 's',
                    color=colors[i % len(colors)], markersize=8)

            # Add trajectory ID labels
            if show_labels:
                mid_point = len(track_data) // 2
                ax.text(track_data['x'].iloc[mid_point],
                        track_data['y'].iloc[mid_point],
                        str(track_id), fontsize=12,
                        color=colors[i % len(colors)],
                        fontweight='bold')

        ax.set_title("trajectories")
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")

        # add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        x_min, x_max = self.df['x'].min(), self.df['x'].max()
        y_min, y_max = self.df['y'].min(), self.df['y'].max()
        margin = 0.5
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        return ax

    def animate_trajectories(self, output_file=None, interval=100, fps=10):
        if self.df is None:
            print("load CSV file first")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        track_ids = self.df['track_id'].unique()
        timestamps = sorted(self.df['timestamp'].unique())

        colors = plt.cm.tab10.colors

        # trajectories history
        trajectory_history = {track_id: [] for track_id in track_ids}

        def update(frame):
            ax.clear()
            current_time = timestamps[frame]

            ax.set_title(f"trajectory animation - TIMESTAMP: {current_time}")

            current_data = self.df[self.df['timestamp'] <= current_time]

            # update trajectory
            for track_id in track_ids:
                track_data = current_data[current_data['track_id'] == track_id]
                if len(track_data) > 0:
                    # save trajectory point
                    trajectory_history[track_id] = list(zip(track_data['x'], track_data['y']))

                    if trajectory_history[track_id]:
                        trajectory = np.array(trajectory_history[track_id])
                        ax.plot(trajectory[:, 0], trajectory[:, 1], '-',
                                color=colors[track_id % len(colors)],
                                alpha=0.7, linewidth=2)
                        # emphasize the last point
                        # ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                        #         color=colors[track_id % len(colors)],
                        #         s=50, label=f"Track {track_id}")

                        # show ID
                        ax.text(trajectory[-1, 0], trajectory[-1, 1] + 0.2,
                                str(track_id), fontsize=10,
                                color=colors[track_id % len(colors)])

            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.grid(True, linestyle='--', alpha=0.7)

            x_min, x_max = self.df['x'].min(), self.df['x'].max()
            y_min, y_max = self.df['y'].min(), self.df['y'].max()
            margin = 0.5
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)

            return ax,

        # init animation
        ani = FuncAnimation(fig, update, frames=len(timestamps),
                            interval=interval, blit=True)

        if output_file:
            ani.save(output_file, writer='ffmpeg', fps=fps)
            print(f"animation saved at: {output_file}")
        else:
            plt.show()

        return ani

    def calculate_statistics(self):
        if self.df is None:
            print("load CSV file first")
            return None

        stats_dict = {}
        track_ids = self.df['track_id'].unique()

        for track_id in track_ids:
            track_data = self.df[self.df['track_id'] == track_id]

            # Calculate trajectory length (total path length)
            points = np.array(list(zip(track_data['x'], track_data['y'])))
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            total_distance = np.sum(distances)

            if 'timestamp' in track_data.columns:
                duration = (track_data['timestamp'].max() - track_data['timestamp'].min()).total_seconds()
            else:
                duration = len(track_data) - 1  # Assume time interval between frames is 1

            # Calculate average speed
            avg_speed = total_distance / max(duration, 1e-6)

            # Calculate maximum speed
            if 'vx' in track_data.columns and 'vy' in track_data.columns:
                speeds = np.sqrt(track_data['vx']**2 + track_data['vy']**2)
                max_speed = speeds.max()
            else:
                max_speed = np.nan

            # Straight-line distance (start to end)
            start_point = points[0]
            end_point = points[-1]
            straight_distance = np.sqrt(np.sum((end_point - start_point)**2))

            # Path efficiency (straight-line distance / actual path length)
            path_efficiency = straight_distance / max(total_distance, 1e-6)

            stats_dict[track_id] = {
                'Track ID': track_id,
                'Point Count': len(track_data),
                'Path Length': total_distance,
                'Duration (sec)': duration,
                'Average Speed': avg_speed,
                'Maximum Speed': max_speed,
                'Straight Distance': straight_distance,
                'Path Efficiency': path_efficiency
            }

        stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')
        return stats_df

    def plot_speed_profile(self, track_id=None):
        if self.df is None:
            print("load CSV file first")
            return

        if 'vx' not in self.df.columns or 'vy' not in self.df.columns:
            print("no velocity data in CSV")
            return

        # Calculate speed magnitude
        self.df['speed'] = np.sqrt(self.df['vx']**2 + self.df['vy']**2)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        if track_id is not None:
            # Plot speed profile for specific trajectory
            track_data = self.df[self.df['track_id'] == track_id]

            if len(track_data) == 0:
                print(f"Track ID {track_id} not found")
                return

            # Time series
            if 'timestamp' in track_data.columns:
                x = (track_data['timestamp'] - track_data['timestamp'].iloc[0]).dt.total_seconds()
                xlabel = "Time (seconds)"
            else:
                x = track_data['frame']
                xlabel = "Frame Number"

            ax.plot(x, track_data['speed'], 'o-', linewidth=2, markersize=4, label=f"Track {track_id}")
            ax.set_title(f"Speed Profile for Track {track_id}")

        else:
            # Plot speed profiles for all trajectories
            track_ids = self.df['track_id'].unique()
            colors = plt.cm.tab10.colors

            for i, tid in enumerate(track_ids):
                track_data = self.df[self.df['track_id'] == tid]

                # Time series
                if 'timestamp' in track_data.columns:
                    x = (track_data['timestamp'] - track_data['timestamp'].iloc[0]).dt.total_seconds()
                    xlabel = "Time (seconds)"
                else:
                    x = track_data['frame']
                    xlabel = "Frame Number"

                ax.plot(x, track_data['speed'], 'o-', color=colors[i % len(colors)],
                        linewidth=2, markersize=4, label=f"Track {tid}")

            ax.set_title("Speed Profiles for All Tracks")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Speed")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')

        return fig, ax

    def plot_heatmap(self, resolution=0.1):
        """Generate trajectory heatmap"""
        if self.df is None:
            print("Please load CSV data first")
            return

        # Create grid
        x_min, x_max = self.df['x'].min(), self.df['x'].max()
        y_min, y_max = self.df['y'].min(), self.df['y'].max()

        x_bins = np.arange(x_min, x_max + resolution, resolution)
        y_bins = np.arange(y_min, y_max + resolution, resolution)

        # Use numpy's histogram2d function to create heatmap data
        heatmap, xedges, yedges = np.histogram2d(
            self.df['x'], self.df['y'], bins=[x_bins, y_bins]
        )

        # Transpose heatmap to match image coordinate system
        heatmap = heatmap.T

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw heatmap
        c = ax.pcolormesh(xedges, yedges, heatmap, cmap='hot', alpha=0.7)
        fig.colorbar(c, ax=ax, label='Point Density')

        # Draw trajectory lines on the heatmap
        self.plot_all_trajectories(show_labels=False, ax=ax, title="Trajectory Heatmap", grid=False)

        return fig, ax

    def export_statistics(self, output_file):
        """Export statistics to CSV file"""
        stats_df = self.calculate_statistics()
        if stats_df is not None:
            stats_df.to_csv(output_file)
            print(f"Statistics exported to: {output_file}")
            return stats_df
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Trajectory Analysis Tool")
    parser.add_argument("--track", type=str, help="Trajectory PKL file path")
    parser.add_argument("--csv", type=str, help="Trajectory CSV file path")
    parser.add_argument("--animate", action="store_true", help="Create trajectory animation")
    parser.add_argument("--output", type=str, default="analysis_results", help="Output directory")
    parser.add_argument("--track_id", type=int, help="Specific track ID to analyze")
    parser.add_argument("--heatmap", action="store_true", help="Generate heatmap")
    parser.add_argument("--speed", action="store_true", help="Generate speed profile")
    parser.add_argument("--stats", action="store_true", help="Calculate statistics")

    args = parser.parse_args()

    # Ensure at least one input file
    if not args.track and not args.csv:
        print("Please specify at least one trajectory file (--track or --csv)")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create analyzer
    analyzer = TrackAnalyzer(track_file=args.track, csv_file=args.csv)

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Default behavior: generate all charts if no options specified
    if not any([args.animate, args.heatmap, args.speed, args.stats]):
        args.animate = args.heatmap = args.speed = args.stats = True

    # Draw trajectory plot
    plt.figure(figsize=(10, 8))
    analyzer.plot_all_trajectories()
    plt.savefig(os.path.join(args.output, f"trajectories_{timestamp}.png"), dpi=300)
    print(f"Trajectory plot saved to: {os.path.join(args.output, f'trajectories_{timestamp}.png')}")

    # Create animation
    if args.animate:
        animation_file = os.path.join(args.output, f"animation_{timestamp}.mp4")
        analyzer.animate_trajectories(output_file=animation_file)

    # Generate heatmap
    if args.heatmap:
        fig, _ = analyzer.plot_heatmap()
        heatmap_file = os.path.join(args.output, f"heatmap_{timestamp}.png")
        fig.savefig(heatmap_file, dpi=300)
        print(f"Heatmap saved to: {heatmap_file}")

    # Generate speed profile
    if args.speed:
        fig, _ = analyzer.plot_speed_profile(args.track_id)
        speed_file = os.path.join(args.output, f"speed_profile_{timestamp}.png")
        fig.savefig(speed_file, dpi=300)
        print(f"Speed profile saved to: {speed_file}")

    # Calculate statistics
    if args.stats:
        stats_file = os.path.join(args.output, f"statistics_{timestamp}.csv")
        analyzer.export_statistics(stats_file)

    # Show all charts
    plt.show()

if __name__ == "__main__":
    main()