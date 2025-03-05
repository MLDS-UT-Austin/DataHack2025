# Instructions

- pip install stuff
- create a github access token and put in `.env` like so:

```
GITHUB_ACCESS_TOKEN=<your_token_here>
```

- run `clone.py` to clone all repos with commits
- run `process.py` to rename all repos with their team number and grade all teams
- copy and paste the schedule from the google sheet into `schedule.csv`
- run `prep_for_judging.py` to generate the judging data and plots