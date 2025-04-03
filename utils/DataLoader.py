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
            dfs.append(df)
        combinedDf = pd.concat(dfs, ignore_index=True)
        combinedDf["timestamp"] = pd.to_datetime(
            {
                "year": combinedDf["y"],
                "month": combinedDf["m"],
                "day": combinedDf["d"],
                "hour": combinedDf["h"],
                "minute": combinedDf["m.1"],  # 分钟列（重命名为m.1）
                "second": combinedDf["s"].astype(float),  # 整数秒部分
            }
        )  # 处理小数秒
        uniqueFrame = combinedDf["timestamp"].unique()
        return combinedDf, uniqueFrame


# if __name__ == "__main__":
#     loader = DataLoader(room="room1", countPerson=1)
#     data, frames = loader.load()
#     print(f"Loaded data with {len(data)} rows and {frames} unique frames")
