import json
import os
from glob import glob

import pandas as pd  # type: ignore

from util import grade

SUBMISSION_DIR = "submissions"  # Directory with cloned repos
OUTPUT_DIR = "output"  # Directory to save the output

ANSWERS_PATH = "answers.csv"

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

answers_df = pd.read_csv(ANSWERS_PATH)

repos = [p for p in glob(f"{SUBMISSION_DIR}/*") if os.path.isdir(p)]
assert repos

teams_list = []

for repo in repos:
    # Load team info
    with open(f"{repo}/submission/team_info.json") as f:
        team_info = json.load(f)
    members_dict = {
        f"{k} {i+1}": v
        for i, m in enumerate(team_info["team_members"])
        for k, v in m.items()
    }

    # Rename the directory to the team number
    new_repo = f"{SUBMISSION_DIR}/team {team_info['team_number']}"
    os.rename(repo, new_repo)
    repo = new_repo

    print(f"grading team {team_info['team_number']}")

    # Grade submission
    submission_df = pd.read_csv(f"{repo}/submission/submission.csv")
    grading_dict = grade(submission=submission_df, answers=answers_df)

    # Append the team info to the teams_df
    new_row = {"team_number": team_info["team_number"], **members_dict, **grading_dict}
    teams_list.append(pd.DataFrame([new_row]))

teams_df = pd.concat(teams_list)
teams_df.sort_values("team_number", inplace=True)

# rank teams on mse and profit
teams_df["mse rank"] = teams_df["mse"].rank(ascending=True).astype(int)
teams_df["profit rank"] = teams_df["profit"].rank(ascending=False).astype(int)

teams_df.to_csv(f"{OUTPUT_DIR}/teams.csv", index=False)
