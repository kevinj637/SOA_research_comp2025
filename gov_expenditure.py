import pandas as pd
import numpy as np
import math


def gov_total_loss_90(file :str) -> (tuple): 
    df = pd.read_csv(f"{file}")
    df.sort_values("Expected Loss Value", axis= 0, ascending=False, inplace=True)
    df_analyze = df.head(int(df.index.size / 10) + df.index.size % 10)
    expected_value = df_analyze.get(["Expected Loss Value"]).sum()
    total_loss_avg = df_analyze.get(["Total Loss Given Failure"]).sum()
    df_analyze.to_csv(f"outlier_{file}")
    cumulative_variance = 0
    print(df_analyze.columns.get_loc("Total Loss Given Failure"))
    for i in range(df_analyze.index.size):
        second_moment = df_analyze.iloc[i, 18] * df_analyze.iloc[i, 22] * df_analyze.iloc[i, 22]
        variance = second_moment - df_analyze.iloc[i, 23] * df_analyze.iloc[i, 23]
        cumulative_variance = variance + cumulative_variance
    standard_deviation = math.sqrt(cumulative_variance)

    print(f"""Region: {df.iloc[3,1]}\nTotal Loss Given Failure: {total_loss_avg.values[0]}\nExpected Value: {expected_value.values[0]}\nStandard Deviation: {standard_deviation}\nVariance: {cumulative_variance}\nNumber of Dams: {df_analyze.index.size}\n\n""")
    return total_loss_avg.values[0], expected_value.values[0], standard_deviation, cumulative_variance, df_analyze.index.size


def gov_average_loss_90(file :str) -> (tuple): 
    df = pd.read_csv(f"{file}")
    df.sort_values("Expected Loss Value", axis= 0, ascending=False, inplace=True)
    df_analyze = df.head(int(df.index.size / 10) + df.index.size % 10)
    expected_value = df_analyze.get(["Expected Loss Value"]).sum()
    total_loss_avg = df_analyze.get(["Total Loss Given Failure"]).sum() / df_analyze.index.size
    df_analyze.to_csv(f"outlier_{file}")
    cumulative_variance = 0
    print(df_analyze.columns.get_loc("Total Loss Given Failure"))
    for i in range(df_analyze.index.size):
        second_moment = df_analyze.iloc[i, 18] * df_analyze.iloc[i, 22] * df_analyze.iloc[i, 22]
        variance = second_moment - df_analyze.iloc[i, 23] * df_analyze.iloc[i, 23]
        cumulative_variance = variance + cumulative_variance
    expected_value = expected_value / df_analyze.index.size
    cumulative_variance = cumulative_variance / df_analyze.index.size
    standard_deviation = math.sqrt(cumulative_variance)

    print(f"""Region: {df.iloc[3,1]}\nAverage Loss Given Failure: {total_loss_avg.values[0]}\nExpected Value: {expected_value.values[0]}\nStandard Deviation: {standard_deviation}\nVariance: {cumulative_variance}\nNumber of Dams: {df_analyze.index.size}\n\n""")
    return total_loss_avg.values[0], expected_value.values[0], standard_deviation, cumulative_variance, df_analyze.index.size

#gov_total_loss_90("dam_data_imputed_flumevale.csv")
#gov_average_loss_90("dam_data_imputed_flumevale.csv")


def total_loss_percentile(file : str, lower_percentile : int, upper_percentile : int) -> (tuple):
    df = pd.read_csv(f"{file}")
    df.sort_values("Expected Loss Value", axis= 0, ascending=True, inplace=True)
    lower_bound = int(df.index.size * lower_percentile / 100)
    upper_bound = int(df.index.size * upper_percentile / 100)
    df_analyze = df.iloc[lower_bound:upper_bound]
    expected_value = df_analyze.get(["Expected Loss Value"]).sum()
    total_loss_avg = df_analyze.get(["Total Loss Given Failure"]).sum()
    df_analyze.to_csv(f"outlier_{file}")
    cumulative_variance = 0
    print(df_analyze.columns.get_loc("Total Loss Given Failure"))
    for i in range(df_analyze.index.size):
        second_moment = df_analyze.iloc[i, 18] * df_analyze.iloc[i, 22] * df_analyze.iloc[i, 22]
        variance = second_moment - df_analyze.iloc[i, 23] * df_analyze.iloc[i, 23]
        cumulative_variance = variance + cumulative_variance
    standard_deviation = math.sqrt(cumulative_variance)

    print(f"""Region: {df.iloc[3,1]}\nTotal Loss Given Failure: {total_loss_avg.values[0]}\nExpected Value: {expected_value.values[0]}\nStandard Deviation: {standard_deviation}\nVariance: {cumulative_variance}\nNumber of Dams: {df_analyze.index.size}\n\n""")
    return total_loss_avg.values[0], expected_value.values[0], standard_deviation, cumulative_variance, df_analyze.index.size

def average_loss_percentile(file : str, lower_percentile : int, upper_percentile : int) -> (tuple):
    df = pd.read_csv(f"{file}")
    df.sort_values("Expected Loss Value", axis= 0, ascending=True, inplace=True)
    lower_bound = int(df.index.size * lower_percentile / 100)
    upper_bound = int(df.index.size * upper_percentile / 100)
    df_analyze = df.iloc[lower_bound:upper_bound]
    #print(df_analyze.to_string())
    expected_value = df_analyze.get(["Expected Loss Value"]).sum()
    total_loss_avg = df_analyze.get(["Total Loss Given Failure"]).sum() / df_analyze.index.size
    df_analyze.to_csv(f"outlier_{file}")
    cumulative_variance = 0
    print(df_analyze.columns.get_loc("Total Loss Given Failure"))
    for i in range(df_analyze.index.size):
        second_moment = df_analyze.iloc[i, 18] * df_analyze.iloc[i, 22] * df_analyze.iloc[i, 22]
        variance = second_moment - df_analyze.iloc[i, 23] * df_analyze.iloc[i, 23]
        cumulative_variance = variance + cumulative_variance
    expected_value = expected_value / df_analyze.index.size
    cumulative_variance = cumulative_variance / df_analyze.index.size
    standard_deviation = math.sqrt(cumulative_variance)

    print(f"""Region: {df.iloc[3,1]}\nAverage Loss Given Failure: {total_loss_avg.values[0]}\nExpected Value: {expected_value.values[0]}\nStandard Deviation: {standard_deviation}\nVariance: {cumulative_variance}\nNumber of Dams: {df_analyze.index.size}\n\n""")
    return total_loss_avg.values[0], expected_value.values[0], standard_deviation, cumulative_variance, df_analyze.index.size

#average_loss_percentile("dam_data_imputed_flumevale.csv", 90, 100)

