import numpy as np
import matplotlib.pyplot as plt
from rescorla_wagner import rescorla_wagner

## Secondary Conditioning

# Learned expectations in secondary conditioning

# Parameters for the Rescorla-Wagner model
epsilon = 0.3  # Learning rate for stimuli (CS1, CS2)
num_trials_pretraining = 100  # Trials in Pre-Training (CS1 only)
num_trials_training = 100     # Trials in Training (CS1 + CS2)
CS2_testing = 1               # CS2 Test Trial

total_trials = num_trials_pretraining + num_trials_training + CS2_testing

# Stimuli presentation and reward arrays for the two training stages (1 = present; 0 = absent)
stimuli_1 = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.zeros(CS2_testing)])     # CS1 always present
stimuli_2 = np.concatenate([np.zeros(num_trials_pretraining), np.ones(num_trials_training), np.ones(CS2_testing)])     # CS2 in second half
rewards = np.concatenate([np.ones(num_trials_pretraining), np.zeros(num_trials_training)])                             # Reward in first half

# Apply Rescorla-Wagner rule
predictions_v, weights_1, weights_2 = rescorla_wagner(stimuli_1, stimuli_2, rewards, epsilon)

# Testing for CS2
testing_2 = weights_1[total_trials - 2] * stimuli_1[total_trials - 1] + weights_2[total_trials - 2] * stimuli_2[total_trials - 1]

# Idealised expectations in secondary conditioning

# Generate 0 to 1 incremental arrays for pre-training and training phases
x_pretraining = np.linspace(0, 1, num_trials_pretraining)
x_training = np.linspace(0, 1, num_trials_training)

# Idealizsd exponential functions for Stimulus 1 (Weights 1)
weights_1_pretraining = (1 - np.exp(-7 * x_pretraining))  # Exponential growth to 1
weights_1_training = np.ones(num_trials_training)         # Maintain a plateau at 1

# Combine both phases for Stimulus 1
idealised_weights_1 = np.concatenate((weights_1_pretraining, weights_1_training))

# Faster growth for Stimulus 2 (Weights 2) so that it predicts the reward by the end of training
weights_2_pretraining = np.zeros(num_trials_pretraining)  # 0 during pre-training
weights_2_training = (1 - np.exp(-7 * x_training))        # Exponential growth to 1

# Combine both phases for Stimulus 2
idealised_weights_2 = np.concatenate((weights_2_pretraining, weights_2_training))

# Summed ideal expectations
summed_ideal_expectations = idealised_weights_1 + idealised_weights_2

# Plot 1: Idealised Expectations for Secondary Conditioning Across Trials
plt.figure(figsize=(10, 5))
plt.plot(idealised_weights_1, label="Weight (Stimulus 1)", color="blue")
plt.plot(idealised_weights_2, label="Weight (Stimulus 2)", color="orange")
plt.plot(summed_ideal_expectations, label="Summed Idealised Expectations", color="green", linestyle="--")
plt.plot(total_trials, idealised_weights_2[-1], 'o', label="Test (Stimulus 2)", color="orange")
plt.xlabel("Trials")
plt.ylabel("Expectations")
plt.title("Idealised Expectations for Secondary Conditioning Across Trials")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Learned Expectations for Secondary Conditioning Across Trials
plt.figure(figsize=(10, 5))
plt.plot(weights_1, label="Weight (Stimulus 1)", color="blue")
plt.plot(weights_2, label="Weight (Stimulus 2)", color="orange")
plt.plot(predictions_v, label="Summed Learned Expectations", color="green", linestyle = "--")
plt.plot((total_trials - 1), testing_2, 'o', label="Test (Stimulus 2)", color="orange")
plt.xlabel("Trials")
plt.ylabel("Expectations")
plt.title("Learned Expectations for Secondary Conditioning Across Trials")
plt.legend()
plt.grid(True)
plt.show()