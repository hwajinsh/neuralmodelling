import numpy as np
import matplotlib.pyplot as plt
from rescorla_wagner import rescorla_wagner

## Blocking

# Parameters for the Rescorla-Wagner model
epsilon = 0.15  # Learning rate for stimuli (CS1, CS2)
num_trials_pretraining = 100  # Trials in Pre-Training (CS1 only)
num_trials_training = 100     # Trials in Training (CS1 + CS2)
#num_trials_result = 50       # Trials in Result phase for visualizing expectations
total_trials = num_trials_pretraining + num_trials_training

# Stimuli presentation arrays for the three stages
stimuli_1 = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training)])
stimuli_2 = np.concatenate([np.zeros(num_trials_pretraining), np.ones(num_trials_training)])

# Reward array (1 = reward, 0 = no reward in the Result Phase)
rewards = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training)])

# Ideal expectations in blocking: reward is always present
ideal_expectations = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training)])

# Apply Rescorla-Wagner rule
predictions_v, weights_1, weights_2 = rescorla_wagner(stimuli_1, stimuli_2, rewards, epsilon)

# Plot 1: Learned predictions, ideal expectations, and stimulus 2
plt.figure(figsize=(10, 5))
plt.plot(predictions_v, label="Learned Predictions", color="blue")
plt.plot(ideal_expectations, label="Ideal Expectations", color="orange")
plt.plot(stimuli_2, label="Stimulus 2", color="green", linestyle="--")
plt.xlabel("Trials")
plt.ylabel("Values")
plt.title("Learned Predictions, Ideal Expectations, and Stimulus 2")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Expectations (Weights) for both stimuli
plt.figure(figsize=(10, 5))
plt.plot(weights_1, label="Weight for Stimulus 1", color="blue")
plt.plot(weights_2, label="Weight for Stimulus 2", color="orange")
plt.xlabel("Trials")
plt.ylabel("Weights")
plt.title("Expectations (Weights) for Stimulus 1 and Stimulus 2")
plt.legend()
plt.grid(True)
plt.show()