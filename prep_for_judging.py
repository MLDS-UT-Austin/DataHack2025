import os

import pandas as pd  # type: ignore
from util import plot

SCHEDULE_PATH = "schedule.csv"
DATA_PATH = "output/teams.csv"
OUTPUT_DIR = "output"  # Directory to save data

OUTPUT_COLS = ["time", "team", "team_name", "repo", "mse", "mse rank", "profit", "profit rank"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

sch_df = pd.read_csv(SCHEDULE_PATH, sep="\t")
data_df = pd.read_csv(DATA_PATH)

# remove empty rows
sch_df = sch_df.dropna(how="all")

# Adjust columns
sch_df.columns = sch_df.columns.str.lower()
data_df = data_df.rename(columns={"team_number": "team"})

# Merge data
sch_df["team"] = sch_df["team"].astype(int)
data_df["team"] = data_df["team"].astype(int)
joined = sch_df.merge(data_df, on="team", how="left")

# Group by room and save to separate files
for room, room_df in joined.groupby("room"):
    plot(room_df, f"{OUTPUT_DIR}/{room}_plot.png")

    # clean up and save
    room_df = room_df[OUTPUT_COLS]
    room_df.columns = (
        room_df.columns.str.replace("_", " ").str.title().str.replace("Mse", "MSE")
    )
    room_df.to_csv(f"{OUTPUT_DIR}/{room}.csv", index=False, sep="\t")
