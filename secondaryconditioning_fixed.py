import numpy as np
import matplotlib.pyplot as plt
from rescorla_wagner import rescorla_wagner

## Secondary Conditioning

# Parameters for the Rescorla-Wagner model
epsilon = 0.4  # Learning rate for stimuli (CS1, CS2)
num_trials_pretraining = 100  # Trials in Pre-Training (CS1 only)
num_trials_training = 100     # Trials in Training (CS1 + CS2)
CS2_testing = 1

total_trials = num_trials_pretraining + num_trials_training + CS2_testing

# Stimuli presentation and reward arrays for the two training stages (1 = present; 0 = absent)
stimuli_1 = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.zeros(CS2_testing)])     # CS1 always present
stimuli_2 = np.concatenate([np.zeros(num_trials_pretraining), np.ones(num_trials_training), np.ones(CS2_testing)])     # CS2 in second half
rewards = np.concatenate([np.ones(num_trials_pretraining), np.zeros(num_trials_training)])                             # Reward in first half

# Apply Rescorla-Wagner rule
predictions_v, weights_1, weights_2 = rescorla_wagner(stimuli_1, stimuli_2, rewards, epsilon)

# Testing
testing_1 = weights_1[total_trials - 2] * stimuli_1[total_trials - 1] + weights_2[total_trials - 2] * stimuli_2[total_trials - 1]

# Plot: Learned predictions and ideal expectations
'''
plt.figure(figsize=(10, 5))
plt.plot(predictions_v, label="Learned Predictions", color="blue")
plt.plot(ideal_expectations, label="Ideal Expectations", color="orange")
plt.xlabel("Trials")
plt.ylabel("Values")
plt.title("Learned Predictions and Ideal Expectations")
plt.legend()
plt.grid(True)
plt.show()
'''

# Plot 2: Expectations (Weights) for both stimuli
plt.figure(figsize=(10, 5))
plt.plot(predictions_v, label="Summed Learned Expectations", color="green", linestyle = "--")
plt.plot(weights_1, label="Weight (Stimulus 1)", color="blue")
plt.plot(weights_2, label="Weight (Stimulus 2)", color="orange")
plt.plot((total_trials - 2), testing_1, 'o', color="orange")
plt.xlabel("Trials")
plt.ylabel("Expectations")
plt.title("Learned Expectations for Secondary Conditioning Across Trials")
plt.legend()
plt.grid(True)
plt.show()