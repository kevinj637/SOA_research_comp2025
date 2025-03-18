import machine_learning
import pandas as pd
import ydf

output = pd.DataFrame()
ydf.verbose(0)
for frequency in range (10):
    flumevale_data = machine_learning.expected_losses_given_min_frequency("dam_data_imputed_flumevale.csv", frequency, 95, False , False)
    lyndrassia_data = machine_learning.expected_losses_given_min_frequency("dam_data_imputed_lyndrassia.csv", frequency, 95, False, False)
    navaldia_data = machine_learning.expected_losses_given_min_frequency("dam_data_imputed_navaldia.csv", frequency, 95, False, False)
    curr_output = pd.DataFrame([flumevale_data | lyndrassia_data | navaldia_data]) #concat
    output = pd.concat([output, curr_output], ignore_index=True)
    print(f"Finished frequency = {frequency} ...")
for frequency in range (10, 50, 5):
    flumevale_data = machine_learning.expected_losses_given_min_frequency("dam_data_imputed_flumevale.csv", frequency, 95, False , False)
    lyndrassia_data = machine_learning.expected_losses_given_min_frequency("dam_data_imputed_lyndrassia.csv", frequency, 95, False, False)
    navaldia_data = machine_learning.expected_losses_given_min_frequency("dam_data_imputed_navaldia.csv", frequency, 95, False, False)
    curr_output = pd.DataFrame([flumevale_data | lyndrassia_data | navaldia_data]) #concat
    output = pd.concat([output, curr_output], ignore_index=True)
    print(f"Finished frequency = {frequency} ...")
output.to_excel("frequencyVSdecreaseloss.xlsx")