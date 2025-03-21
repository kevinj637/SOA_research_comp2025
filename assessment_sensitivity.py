import pandas as pd
import numpy as np
import ydf
import matplotlib.pyplot as plot
import math
import scipy.stats
import gov_expenditure

def train_model(df : pd.DataFrame):
    train_df = df.copy(True).drop(columns=["ID", "Years Modified", "Assessment Date", "Assessment Date","Loss given failure - prop (Qm)","Loss given failure - liab (Qm)", "Loss given failure - BI (Qm)", "Total Loss Given Failure", "Expected Loss Value", "Hazard"])
    model = ydf.GradientBoostedTreesLearner(label="Probability of Failure", task=ydf.Task.REGRESSION).train(train_df)
    #returns model
    return model

def predict_model(file : str, model : ydf.GenericModel, assessment : str, replace_all : bool):
    df = pd.read_csv(file)
    test_df = df.copy(False).drop(columns=["ID", "Years Modified", "Assessment Date", "Assessment Date","Loss given failure - prop (Qm)","Loss given failure - liab (Qm)", "Loss given failure - BI (Qm)", "Total Loss Given Failure", "Expected Loss Value", "Hazard"])
    if replace_all:
        for i in range(len(test_df)):
            test_df.loc[i, "Assessment"] = assessment
        print("\n\n Replaced_ALL \n\n")
    else:
        for i in range(len(test_df)):
            if test_df.loc[i, "Assessment"] == "Not Rated" or test_df.loc[i, "Assessment"] == "Not Available":
                test_df.loc[i, "Assessment"] = assessment
    new_probabilities = model.predict(test_df)
    new_df = df.copy(True)
    new_df["Probability of Failure"] = pd.Series(new_probabilities)
    for i in range(new_df.index.size):
        new_df.loc[i, "Expected Loss Value"] = new_df.loc[i, "Probability of Failure"] * new_df.loc[i, "Total Loss Given Failure"] 
    #returns new file name
    new_df.to_csv(f"machine_learning_assessment_adjusted_{df.loc[0, "Region"]}_Replaced_All_is_{replace_all}.csv", index = False)
    return new_df

def draw_graph_And_statistics(old_df : pd.DataFrame, new_df : pd.DataFrame, export : bool, replace_all : bool, assessment : str):
    compare_new = new_df["Expected Loss Value"].to_numpy()
    compare_old = old_df["Expected Loss Value"].to_numpy()
    plot.hist([compare_old, compare_new], bins=50,label=["Original", "After Frequency Change"])
    plot.xlabel("Expected Loss Value, millions(£Q)")
    plot.ylabel("Number of Dams")
    plot.title(f"Reduction to losses by change in assessment in {old_df.loc[0, "Region"]}")
    plot.legend()
    if export:
        plot.savefig(f"assessment_adjusted_expected_loss__{old_df.loc[0, "Region"]}_histogram_{assessment}_Replaced_All_is_{replace_all}.png")
    plot.clf()

    new_data = gov_expenditure.yearly_loss_percentile(f"machine_learning_assessment_adjusted_{old_df.loc[0, "Region"]}_Replaced_All_is_{replace_all}.csv", 0, 100, False)
    old_data = gov_expenditure.yearly_loss_percentile("dam_data_imputed_flumevale.csv", 0, 100, False)
    gov_threshold = scipy.stats.norm.ppf(95/100) * old_data[2] + old_data[1] #one standard deviation, covers 95% to 99.7% of all cases
    gov_reserve = scipy.stats.norm.ppf(0.997) * old_data[2] + old_data[1] - gov_threshold
    new__gov_detachment_point = scipy.stats.norm.ppf(0.997) * new_data[2] + new_data[1] #the amount of money for 99.7% of all cases
    new_gov_threshold = new__gov_detachment_point - gov_reserve
    new_percentile = scipy.stats.norm.cdf(new_gov_threshold, new_data[1], new_data[2])
    if True:
        print("Based on the original data, the threshold should be:", gov_threshold, "and the reserve is", gov_reserve)
        print(f"The expected value of the new data is {new_data[1]} with standard deviation {new_data[2]}.")
        print(f"This means the government detachment point is {new__gov_detachment_point} and the detachment point is {new_gov_threshold}.")
        print(f"The insurance companies will expect to pay {new_data[1]} instead of {old_data[1]}. This represents a decreased payout of {new_data[1] - old_data[1]}")
        print(f"The government threshold will also shift by {new_gov_threshold - gov_threshold}")
        print(f"This is the {new_percentile * 100} percentile.")

    ans = {f"Threshold percent {old_df.loc[0,"Region"]}" : new_percentile}
    ans.update({f"Original Threshold {old_df.loc[0,"Region"]}": gov_threshold})
    ans.update({f"New Threshold {old_df.loc[0,"Region"]}" : new_gov_threshold})
    ans.update({f"Change in Threshold {old_df.loc[0,"Region"]}" : new_gov_threshold - gov_threshold})
    ans.update({f"Original Expected Payout {old_df.loc[0,"Region"]}" : old_data[1]})
    ans.update({f"New Expected Payout {old_df.loc[0,"Region"]}" : new_data[1]})
    ans.update({f"Change in Payout {old_df.loc[0,"Region"]}" : new_data[1] - old_data[1]})
    return ans

def find_assessment_sensitivity_region(file : str, replace_all : bool, export : bool):
    old_df = pd.read_csv(file)
    model = train_model(old_df)
    new_df = predict_model(file, model, "Poor", replace_all)
    ans = pd.DataFrame([draw_graph_And_statistics(old_df, new_df, export, replace_all, "Poor")])
    new_df = predict_model(file, model, "Unsatisfactory", replace_all)
    ans = pd.concat([ans, pd.DataFrame([draw_graph_And_statistics(old_df, new_df, export, replace_all, "Unsatisfactory")])])
    new_df = predict_model(file, model, "Fair", replace_all)
    ans = pd.concat([ans, pd.DataFrame([draw_graph_And_statistics(old_df, new_df, export, replace_all, "Fair")])])
    new_df = predict_model(file, model, "Satisfactory", replace_all)
    ans = pd.concat([ans, pd.DataFrame([draw_graph_And_statistics(old_df, new_df, export, replace_all, "Satisfactory")])])
    return ans