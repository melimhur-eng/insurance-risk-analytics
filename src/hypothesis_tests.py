import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind


def create_claim_indicator(df, claims_col="TotalClaims"):
    """
    Create binary claim occurrence column.
    """
    df = df.copy()

    df["HasClaim"] = np.where(df[claims_col] > 0, 1, 0)

    return df


def calculate_margin(df):
    """
    Create margin column.
    """
    df = df.copy()

    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    return df


def calculate_loss_ratio(df):
    """
    Create loss ratio safely.
    """
    df = df.copy()

    df["LossRatio"] = np.where(
        df["TotalPremium"] > 0,
        df["TotalClaims"] / df["TotalPremium"],
        0
    )

    return df


def chi_square_test(df, group_col, target_col):
    """
    Chi-square test for categorical variables.
    """

    contingency_table = pd.crosstab(
        df[group_col],
        df[target_col]
    )

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return {
        "chi2_statistic": chi2,
        "p_value": p,
        "degrees_of_freedom": dof,
        "expected_freq": expected
    }


def t_test(group_a, group_b):
    """
    Independent t-test.
    """

    statistic, p_value = ttest_ind(
        group_a,
        group_b,
        equal_var=False,
        nan_policy="omit"
    )

    return {
        "t_statistic": statistic,
        "p_value": p_value
    }


def hypothesis_decision(p_value, alpha=0.05):
    """
    Make hypothesis decision.
    """

    if p_value < alpha:
        return "Reject Null Hypothesis"

    return "Fail to Reject Null Hypothesis"


def summarize_results(
    hypothesis,
    test_used,
    p_value,
    alpha=0.05
):
    """
    Create formatted summary dictionary.
    """

    decision = hypothesis_decision(
        p_value,
        alpha
    )

    return {
        "Hypothesis": hypothesis,
        "Test": test_used,
        "P-Value": round(p_value, 5),
        "Decision": decision
    }