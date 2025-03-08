import json
import os
import subprocess
from glob import glob

import pandas as pd  # type: ignore

from util import grade

SUBMISSION_DIR = "submissions"  # Directory with cloned repos
OUTPUT_DIR = "output"  # Directory to save the output

WIND_SPEED_ANSWERS_PATH = "example_dataset_with_submission_damage/submission.csv"
DAMAGES_PATH = "example_dataset_with_submission_damage/damages.csv"

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

wind_speed_answers_df = pd.read_csv(WIND_SPEED_ANSWERS_PATH)
damages_df = pd.read_csv(DAMAGES_PATH)

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
    if repo != new_repo:
        assert not os.path.exists(new_repo)
    os.rename(repo, new_repo)
    repo = new_repo

    print(f"grading team {team_info['team_number']}")

    # Grade submission
    submission_df = pd.read_csv(f"{repo}/submission/submission.csv")
    grading_dict = grade(
        submission=submission_df,
        wind_speed_answers=wind_speed_answers_df,
        damages=damages_df,
    )

    # add GitHub repo link (use git origin remote)
    try:
        # Get the git origin URL
        git_remote = subprocess.check_output(
            ["git", "-C", repo, "config", "--get", "remote.origin.url"], text=True
        ).strip()
        repo_link = git_remote
    except Exception as e:
        print(f"Failed to get git remote for {repo}: {e}")
        repo_link = ""

    # Append the team info to the teams_df
    new_row = {
        "team_number": team_info["team_number"],
        "team_name": team_info["team_name"],
        "repo": repo_link,
        **members_dict,
        **grading_dict,
    }
    teams_list.append(pd.DataFrame([new_row]))

teams_df = pd.concat(teams_list)
teams_df.sort_values("team_number", inplace=True)

# rank teams on mse and profit
teams_df["mse rank"] = teams_df["mse"].rank(ascending=True, method="min").astype(int)
teams_df["profit rank"] = (
    teams_df["profit"].rank(ascending=False, method="min").astype(int)
)

teams_df.to_csv(f"{OUTPUT_DIR}/teams.csv", index=False)
