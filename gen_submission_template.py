import os
import re
from glob import glob

import pandas as pd

ANSWERS_PATH = "7am_run/submission.csv"
EVENT_PATHS = "7am_run/events_students/event_*.csv"
SAVE_PATH = "../DataHack2025/submission/submission.csv"

CITY="GANopolis"

df = pd.read_csv(ANSWERS_PATH)
df = df.astype("object")
df.iloc[:, 1:] = pd.NA


df["price"] = df["event_number"].apply(lambda x: f"<price for event {x} in {CITY}>")

event_ends = []
for event_path in glob(EVENT_PATHS):
    event_id = re.match(r".*event_(\d+).csv", event_path).group(1)
    event_df = pd.read_csv(event_path)
    end = event_df["hour"].max().item()
    event_ends.append((event_id, end))

event_ends.sort(key=lambda x: int(x[0]))
event_ends = dict(event_ends)

# <wind_speed for event 1 at hour 8095>
df["0"] = df["event_number"].apply(
    lambda x: f"<wind_speed for event {x} in {CITY} at hour {event_ends[str(x)]+1}>"
)
df["1"] = df["event_number"].apply(
    lambda x: f"<wind_speed for event {x} in {CITY} at hour {event_ends[str(x)]+2}>"
)

df.to_csv(SAVE_PATH, index=False)

with open(SAVE_PATH, "r") as f:
    content = f.read()

content = re.sub(r",,+", ", ...", content)

with open(SAVE_PATH, "w") as f:
    f.write(content)
