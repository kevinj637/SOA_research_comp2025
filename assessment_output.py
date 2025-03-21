import assessment_sensitivity
import pandas as pd

flumevale_pd = assessment_sensitivity.find_assessment_sensitivity_region("dam_data_imputed_flumevale.csv", False, True)
lyndrassia_pd = assessment_sensitivity.find_assessment_sensitivity_region("dam_data_imputed_lyndrassia.csv", False, True)
navaldia_pd = assessment_sensitivity.find_assessment_sensitivity_region("dam_data_imputed_navaldia.csv", False, True)

output_no_replace = pd.concat([flumevale_pd, lyndrassia_pd, navaldia_pd])
output_no_replace.to_excel("assessmentVSdecreaselossReplaceSOME.xlsx")