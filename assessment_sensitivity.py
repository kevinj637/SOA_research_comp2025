import pandas as pd
import numpy as np
import ydf

df = pd.read_csv("dam_data_imputed_flumevale.csv")
df["Assessment"].value_counts()
train_df = df.copy()
#train_df.columns
train_df = train_df.drop(columns=["ID", "Years Modified", "Assessment Date", "Assessment Date","Loss given failure - prop (Qm)","Loss given failure - liab (Qm)", "Loss given failure - BI (Qm)", "Total Loss Given Failure", "Expected Loss Value"])

test_df = df.copy()
model = ydf.GradientBoostedTreesLearner(label="Probability of Failure", task=ydf.Task.REGRESSION)
evaluation = model.cross_validation(test_df, folds = 10)
evaluation