from scipy.stats import pearsonr
import pandas as pd

from visualization.plot_metrics.plot_helpers import plot_correlation


def calculate_correlations(df_full, rrr_df, metric_to_correlate="test_accuracy"):
    """
    Calculate the correlation between the metric and the other columns in the dataframe.
    """

    grouped_df = df_full.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    # reorder the df that the columns are the metrics and the rows are the methods
    grouped_df = grouped_df.pivot(index="Method", columns="Metric", values="Value")

    # join the rrr_df with the grouped_df
    rrr_df = rrr_df.set_index("Method")

    rrr_df = rrr_df[[metric_to_correlate]]

    # join
    corr = grouped_df.join(rrr_df, how="inner")
    corr = corr.corr()[metric_to_correlate].sort_values(ascending=False)
    # drop the test_accuracy column
    corr = corr.drop(index=metric_to_correlate)

    # drop nan cols
    corr = corr.dropna()

    return corr


def calculate_correlations_with_significance(
    df_full, rrr_df, metric_to_correlate="test_accuracy"
):
    grouped_df = df_full.groupby(["Method", "Metric"])["Value"].mean().reset_index()
    grouped_df = grouped_df.pivot(index="Method", columns="Metric", values="Value")

    rrr_df = rrr_df.set_index("Method")
    rrr_df = rrr_df[[metric_to_correlate]]
    corr_df = grouped_df.join(rrr_df, how="inner")

    # Prepare a DataFrame to hold results
    results = pd.DataFrame(columns=["Correlation", "P-value"])

    # Iterate through the columns to calculate correlation and p-value
    for column in corr_df.columns[
        :-1
    ]:  # Exclude the last column which is 'metric_to_correlate'
        if column != metric_to_correlate:  # Just in case the order changes in future
            correlation, p_value = pearsonr(
                corr_df[column].dropna(), corr_df[metric_to_correlate].dropna()
            )
            results.loc[column] = [correlation, p_value]

    """
    A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, suggesting that an observed correlation is statistically significant.
    A large p-value (> 0.05) suggests that the observed correlation could have occurred by chance, and thus, evidence against the null hypothesis is weak.
    
    """

    results = results.sort_values(by="Correlation", ascending=False)
    results = results.dropna()

    results["Significant"] = results["P-value"] < 0.05
    results["P-value Text"] = results["P-value"].apply(lambda p: f"p={p:.3f}")

    return results


def calc_and_plot_correlation(
    df_full,
    rrr_df,
    metric_to_correlate="test_accuracy",
    visualization_save_dir=None,
    title_prefix="",
):
    corr = calculate_correlations_with_significance(
        df_full, rrr_df, metric_to_correlate=metric_to_correlate
    )
    plot_correlation(
        corr,
        visualization_save_dir=visualization_save_dir,
        title=f"{title_prefix}Correlation of metrics with {metric_to_correlate}",
    )
