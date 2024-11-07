import numpy as np
import matplotlib.pyplot as plt
from rescorla_wagner import rescorla_wagner
from rescorla_wagner_overshadowing import rescorla_wagner_overshadowing

## Overshadowing

# Parameters for the Rescorla-Wagner model
epsilon = 0.15
num_trials_training = 100     # Trials in Training (CS1 + CS2)
CS1_testing = 1
CS2_testing = 1      
total_trials = num_trials_training + CS1_testing + CS2_testing

# Stimuli presentation arrays for the three stages
stimuli_1 = np.concatenate([np.ones(num_trials_training), np.ones(CS1_testing), np.zeros(CS2_testing)])
stimuli_2 = np.concatenate([np.ones(num_trials_training), np.zeros(CS1_testing), np.ones(CS2_testing)])

# Reward array (1 = reward, 0 = no reward in the Result Phase)
rewards = np.concatenate([np.ones(num_trials_training)])

# Apply Rescorla-Wagner rule
predictions_v, weights_1, weights_2 = rescorla_wagner(stimuli_1, stimuli_2, rewards, epsilon)

# Testing
testing_1 = weights_1[total_trials - 3] * stimuli_1[total_trials - 2] + weights_2[total_trials - 3] * stimuli_2[total_trials - 2]
testing_2 = weights_1[total_trials - 3] * stimuli_1[total_trials - 1] + weights_2[total_trials - 3] * stimuli_2[total_trials - 1]

# Ideal expectations in overshadowing: RW model correctly reflects the idealised expectations
ideal_expectations = predictions_v
ideal_weight_1 = weights_1
ideal_weight_2 = weights_2
ideal_test_1 = testing_1
ideal_test_2 = testing_2

# Plot 1: Idealised Expectations for Overshadowing Across Trials
plt.figure(figsize=(10, 5))
plt.plot(predictions_v, label="Summed Idealised Expectations", color="green", linestyle = "--")
plt.plot(ideal_weight_1, label="Ideal Expectations (Stimulus 1)", color="blue")
plt.plot(ideal_weight_2, label="Ideal Expectations (Stimulus 2)", color="orange")
plt.plot((total_trials - 2), ideal_test_1, 'o', color="blue")
plt.plot((total_trials - 1), ideal_test_2, 'o', color="orange")
plt.xlabel("Trials")
plt.ylabel("Expectations")
plt.title("Idealised Expectations for Overshadowing Across Trials")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Learned Expectations for Overshadowing Across Trials
plt.figure(figsize=(10, 5))
plt.plot(weights_1, label="Weight (Stimulus 1)", color="blue")
plt.plot(weights_2, label="Weight (Stimulus 2)", color="orange")
plt.plot(predictions_v, label="Summed Learned Expectations", color="green", linestyle = "--")
plt.plot((total_trials - 2), testing_1, 'o', color="blue")
plt.plot((total_trials - 1), testing_2, 'o', color="orange")
plt.xlabel("Trials")
plt.ylabel("Expectations")
plt.title("Learned Expectations for Overshadowing Across Trials")
plt.legend()
plt.grid(True)
plt.show()

# The weights of both simuli are the same since the same learning rate (alpha) is given to both stimuli
# Modification of RW rule with two different learning rates needed to plot different weight curves for S1 and S2

## EXTRA: we can implement another RW code to include 2 learning rates
epsilon_1 = 0.15
epsilon_2 = 0.3

predictions_v_new, weights_1_new, weights_2_new = rescorla_wagner_overshadowing(stimuli_1, stimuli_2, rewards, epsilon_1, epsilon_2)

# Testing 
testing_1_new = weights_1_new[total_trials - 3] * stimuli_1[total_trials - 2] + weights_2_new[total_trials - 3] * stimuli_2[total_trials - 2]
testing_2_new = weights_1_new[total_trials - 3] * stimuli_1[total_trials - 1] + weights_2_new[total_trials - 3] * stimuli_2[total_trials - 1]

# Plot 3: Expectations (Weights) for both stimuli with different weight
plt.figure(figsize=(10, 5))
plt.plot(weights_1_new, label="Weight (Stimulus 1)", color="blue")
plt.plot(weights_2_new, label="Weight (Stimulus 2)", color="orange")
plt.plot(predictions_v, label="Summed Learned Expectations", color="green", linestyle = "--")
plt.plot((total_trials - 2), testing_1_new, 'o', color="blue")
plt.plot((total_trials - 1), testing_2_new, 'o', color="orange")
plt.xlabel("Trials")
plt.ylabel("Expectations")
plt.title("Learned Expectations for Overshadowing Across Trials (with Varying Learning Rates)")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show() 