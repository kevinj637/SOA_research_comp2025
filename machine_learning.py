import pandas as pd
import numpy as np
import ydf
import matplotlib.pyplot as plot
import gov_expenditure
import scipy.stats

def expected_losses_given_min_frequency(file : str, frequency : int, initial_percentile: float, make_graph : bool, verbose : bool) -> tuple:
    #obtain probabilities
    df = pd.read_csv(file)
    train_df = df.copy(True)
    #train_df.columns
    train_df = train_df.drop(columns=["ID", "Years Modified", "Assessment Date", "Assessment Date","Loss given failure - prop (Qm)","Loss given failure - liab (Qm)", "Loss given failure - BI (Qm)", "Total Loss Given Failure", "Expected Loss Value"])
    test_df = train_df.copy(True)
    for i in range(test_df.index.size):
        if (test_df.loc[i, "Inspection Frequency"] < frequency):
            test_df.loc[i, "Inspection Frequency"] = frequency
    model = ydf.GradientBoostedTreesLearner(label="Probability of Failure", task=ydf.Task.REGRESSION).train(train_df)
    ydf_prediction = model.predict(test_df)
    for i in range(test_df.index.size):
        test_df.loc[i, "Probability of Failure"] = ydf_prediction[i]
    
    #get new numbers
    new_df = df.copy(True) 
    new_df = new_df.replace(new_df["Probability of Failure"], pd.Series(ydf_prediction))
    for i in range(new_df.index.size):
        new_df.loc[i, "Expected Loss Value"] = test_df.loc[i, "Probability of Failure"] * new_df.loc[i, "Total Loss Given Failure"] 
    new_df.to_csv(f"machine_learning_frequency_adjusted_{df.loc[0, "Region"]}.csv", index = False)

    if make_graph:
        compare_new = new_df["Expected Loss Value"].to_numpy()
        compare_old = df["Expected Loss Value"].to_numpy()
        plot.hist([compare_old, compare_new], bins=50,label=["Original", "After Frequency Change"])
        plot.xlabel("Expected Loss Value, millions(£Q)")
        plot.ylabel("Number of Dams")
        plot.title(f"Reduction to losses by increased inspection frequency in {df.loc[0, "Region"]}")
        plot.legend()
        plot.savefig(f"frequency_adjusted_expected_loss__{df.loc[0, "Region"]}_histogram.png")

    new_data = gov_expenditure.total_loss_percentile(f"machine_learning_frequency_adjusted_{df.loc[0, "Region"]}.csv", 0, 100, False)
    old_data = gov_expenditure.total_loss_percentile(file, 0, 100, False)
    gov_threshold = scipy.stats.norm.ppf(initial_percentile/100) * old_data[2] + old_data[1] #one standard deviation, covers 95% to 99.7% of all cases
    gov_reserve = scipy.stats.norm.ppf(0.997) * old_data[2] + old_data[1] - gov_threshold
    new__gov_detachment_point = scipy.stats.norm.ppf(0.997) * new_data[2] + new_data[1] #the amount of money for 99.7% of all cases
    new_gov_threshold = new__gov_detachment_point - gov_reserve
    new_percentile = scipy.stats.norm.cdf(new_gov_threshold, new_data[1], new_data[2])
    print("Based on the original data, the threshold should be:", gov_threshold, "and the reserve is", gov_reserve)
    print(f"The expected value of the new data is {new_data[1]} with standard deviation {new_data[2]}.")
    print(f"This means the government detachment point is {new__gov_detachment_point} and the detachment point is {new_gov_threshold}.")
    print(f"The insurance companies will expect to pay {new_data[1]} instead of {old_data[1]}. This represents a decreased payout of {new_data[1] - old_data[1]}")
    print(f"The government threshold will also shift by {new_gov_threshold - gov_threshold}")
    print(f"This is the {new_percentile * 100} percentile.")
    
    return new_percentile