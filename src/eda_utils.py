import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(df, column):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column],
        kde=True
    )

    plt.title(f"Distribution of {column}")

    plt.show()


def plot_boxplot(df, column):

    plt.figure(figsize=(8, 5))

    sns.boxplot(x=df[column])

    plt.title(f"Boxplot of {column}")

    plt.show()


def plot_loss_ratio(df, category):

    grouped = (
        df.groupby(category)[
            ["TotalClaims", "TotalPremium"]
        ].sum()
    )

    grouped["LossRatio"] = (
        grouped["TotalClaims"] /
        grouped["TotalPremium"]
    )

    grouped["LossRatio"].sort_values().plot(
        kind="bar",
        figsize=(12, 5)
    )

    plt.ylabel("Loss Ratio")

    plt.title(f"Loss Ratio by {category}")

    plt.show()