import numpy as np
import matplotlib.pyplot as plt
from rescorla_wagner import rescorla_wagner

## Explaining Away

# Parameters for the Rescorla-Wagner model
epsilon = 0.15
num_trials_pretraining = 100 
num_trials_training = 100    
CS1_testing = 1
CS2_testing = 1      
total_trials = num_trials_pretraining + num_trials_training + CS1_testing + CS2_testing


stimuli_1 = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.ones(CS1_testing), np.zeros(CS2_testing)])
stimuli_2 = np.concatenate([np.ones(num_trials_pretraining), np.zeros(num_trials_training), np.zeros(CS1_testing), np.ones(CS2_testing)])


# Reward array (1 = reward, 0 = no reward in the Result Phase)
rewards = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training)])

# Apply Rescorla-Wagner rule
predictions_v, weights_1, weights_2 = rescorla_wagner(stimuli_1, stimuli_2, rewards, epsilon)

# Testing
testing_1 = weights_1[total_trials - 3] * stimuli_1[total_trials - 2] + weights_2[total_trials - 3] * stimuli_2[total_trials - 2]
testing_2 = weights_1[total_trials - 3] * stimuli_1[total_trials - 1] + weights_2[total_trials - 3] * stimuli_2[total_trials - 1]

#CHANGEEEEEEEEEEEE
# Ideal expectations in blocking: RW model correctly reflects the idealised expectations
ideal_expectations = predictions_v
ideal_weight_1 = weights_1
ideal_weight_2 = weights_2
ideal_test_1 = testing_1
ideal_test_2 = testing_2

# Plot 1: Idealised Expectations for Explaining Away Across Trials
plt.figure(figsize=(10, 5))
plt.plot(predictions_v, label="Summed Idealised Expectations", color="green", linestyle = "--")
plt.plot(ideal_weight_1, label="Ideal Expectations (Stimulus 1)", color="blue")
plt.plot(ideal_weight_2, label="Ideal Expectations (Stimulus 2)", color="orange")
plt.plot((total_trials - 2), ideal_test_1, 'o', color="blue")
plt.plot((total_trials - 1), ideal_test_2, 'o', color="orange")
plt.xlabel("Trials")
plt.ylabel("Values")
plt.title("Idealised Expectations for Explaining Away Across Trials")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Learned Expectations for Explaining Away Across Trials
plt.figure(figsize=(10, 5))
plt.plot(predictions_v, label="Summed Learned Expectations", color="green", linestyle = "--")
plt.plot(weights_1, label="Weight (Stimulus 1)", color="blue")
plt.plot(weights_2, label="Weight (Stimulus 2)", color="orange")
plt.plot((total_trials - 2), testing_1, 'o', color="blue")
plt.plot((total_trials - 1), testing_2, 'o', color="orange")
plt.xlabel("Trials")
plt.ylabel("Weights")
plt.title("Learned Expectations for Explaining Away Across Trials")
plt.legend()
plt.grid(True)
plt.show()