# DataHack2025

Official repository for DataHack 2025

## Getting Started

1. Join the [Discord Server](https://discord.gg/qKmmB35TM8) for resources, announcements, questions, etc.
2. Add your teammates to this repo (Settings -> Collaborators -> Add people)
3. For details on how to use GitHub/Git, click [here](https://medium.com/@lahari.kethinedi/github-a-step-by-step-guide-from-clone-to-push-ffad25b6313a)

## File Structure

- `data/`
  - `event_1.csv` Data for the first 120 hours of event 1
  - ...
  - `event_10.csv` Data for the first 120 hours of event 10
  - `training_data.csv` 6 months of training data
- `submission/` see 'Submission' for more details
  - `slides.pptx`
  - `submission.csv`
  - `team_info.json`
  - *Note: please do not add/remove any files in this directory!*
- `README.md`
- `<any code you write>`
- `<any folders you add>`

## Submission

Please update, commit, ***and push*** the following files to your GitHub Classroom repo:

**Due at 3:30 pm:**

- `submission/team_info.json`: Team name, number, and members
- `submission/submission.csv`: Prices and wind speed predictions for each event in GANopolis. You are predicting wind speeds for the second 120 hours of each event.
- *Note: you are only submitting for GANopolis, all other locations are solely features.*

**Due at 4:00 pm:**

- `submission/slides.pptx`: Presentation slides
- Code: Please submit what you used to analyze the data and generate your `submission/submission.csv` file. This should be well commented/readable.
- *Note: your repo will lock at 4:00 pm, so please push your final slides/code before then!*
