import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore


def get_profit(
    wind_speed: pd.Series, prices: pd.Series, damages: pd.Series
) -> pd.Series:
    assert (prices >= 0).all()
    demand = 10000 - (10000 / 500) * prices
    unit_cm = prices - damages
    profit = demand * unit_cm
    return profit


def grade(
    submission: pd.DataFrame, wind_speed_answers: pd.DataFrame, damages: pd.DataFrame
) -> dict:
    hour_cols = [s for s in wind_speed_answers.columns if s.lstrip("-").isdigit()]

    for col in hour_cols:
        assert col in submission.columns
    assert len(submission) == len(wind_speed_answers)

    # Calculate the mean squared error for each wind speed column
    mses = submission[hour_cols].sub(wind_speed_answers[hour_cols]).pow(2).mean(axis=1)

    output = {f"mse_{i}": mse for i, mse in enumerate(mses)}
    output |= {"mse": mses.mean()}

    # Get profit from the submission
    total_damages = damages[hour_cols].sum(axis=1)
    profits = get_profit(
        wind_speed=wind_speed_answers["wind_speed"],
        prices=submission["price"],
        damages=total_damages,
    )

    output |= {f"profit_{i}": profit for i, profit in enumerate(profits)}
    output |= {"profit": profits.sum()}

    return output


def plot(df: pd.DataFrame, savepath: str):
    df["team"] = df["team"].astype(int)

    # Define the columns for MSE and Profit plots
    mse_cols = [f"mse_{i}" for i in range(10)]
    profit_cols = [f"profit_{i}" for i in range(10)]

    df = df.dropna(subset=mse_cols + profit_cols)

    # Melt the data into long format for plotting
    df_mse = df.melt(
        id_vars=["team"], value_vars=mse_cols, var_name="round", value_name="val"
    )
    df_profit = df.melt(
        id_vars=["team"], value_vars=profit_cols, var_name="round", value_name="val"
    )

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Plot MSE: Box plot
    sns.boxplot(
        x="team",
        y="val",
        data=df_mse,
        ax=axes[0],
        # color="lightgray",
        width=0.2,
    )
    axes[0].set_title(f"MSE Across Events")
    axes[0].set_xlabel("Team Number")
    axes[0].set_ylabel("MSE")

    # Plot Profit: Box plot
    sns.boxplot(
        x="team",
        y="val",
        data=df_profit,
        ax=axes[1],
        # color="lightgray",
        width=0.2,
    )
    axes[1].set_title(f"Profit Across Events")
    axes[1].set_xlabel("Team Number")
    axes[1].set_ylabel("Profit")

    # Set an overall title and adjust the layout
    plt.tight_layout()

    # Save the plot with the room number in the file name
    plt.savefig(savepath)
    plt.close()
