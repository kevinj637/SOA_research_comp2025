import machine_learning

frequency = int(input())
percent1 = machine_learning.expected_losses_given_min_frequency("dam_data_imputed_flumevale.csv", frequency, 95, False , False)
percent2 = machine_learning.expected_losses_given_min_frequency("dam_data_imputed_lyndrassia.csv", frequency, 95, False, False)
percent3 = machine_learning.expected_losses_given_min_frequency("dam_data_imputed_navaldia.csv", frequency, 95, False, False)