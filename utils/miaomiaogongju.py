import pandas as pd
import glob
import os


class DataLoader:

    def __init__(self, room, countPerson):
        self.minFrame = 1000
        baseDir = "people-gait/"
        subPath = "fixed_route/77Ghz_radar/"
        self.targetDir = os.path.join(baseDir, room, str(countPerson), subPath, "*.csv")
        self.filePaths = glob.glob(self.targetDir)

    def load(self):
        dfs = []
        for file in self.filePaths:
            df = pd.read_csv(file)
            # Create timestamp column
            df["timestamp"] = pd.to_datetime(
                {
                    "year": df["y"],
                    "month": df["m"],
                    "day": df["d"],
                    "hour": df["h"],
                    "minute": df["m.1"],  # minute column (renamed to m.1)
                    "second": df["s"].astype(float),  # integer seconds part
                }
            )  # handle decimal seconds
            # Remove original datetime columns
            df = df.drop(columns=["y", "m", "d", "h", "m.1", "s"])
            dfs.append(df)
        combinedDf = pd.concat(dfs, ignore_index=True)
        uniqueFrame = combinedDf["timestamp"].unique()
        return combinedDf, uniqueFrame


if __name__ == "__main__":
    loader = DataLoader(room="room1", countPerson=1)
    data, frames = loader.load()
    print(f"Loaded data with {len(data)} rows and {frames} unique frames")
    data
