import numpy as np
import matplotlib.pyplot as plt
from rescorla_wagner import rescorla_wagner

## Secondary Conditioning

# Parameters for the Rescorla-Wagner model
epsilon = 0.4  # Learning rate for stimuli (CS1, CS2)
num_trials_pretraining = 100  # Trials in Pre-Training (CS1 only)
num_trials_training = 100     # Trials in Training (CS1 + CS2)
total_trials = num_trials_pretraining + num_trials_training 

# Stimuli presentation and reward arrays for the two training stages 
stimuli_1 = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training)])     # CS1 always present
stimuli_2 = np.concatenate([np.zeros(num_trials_pretraining), np.ones(num_trials_training)])    # CS2 in second half
rewards = np.concatenate([np.ones(num_trials_pretraining), np.zeros(num_trials_training)])      # Reward in first half

# Ideal Expectation of Rewards
### Is this supposed to be ones? 
ideal_expectations = np.ones(total_trials)

# Apply Rescorla-Wagner rule
predictions_v, weights_1, weights_2 = rescorla_wagner(stimuli_1, stimuli_2, rewards, epsilon)

# Plot: Learned predictions and ideal expectations
plt.figure(figsize=(10, 5))
plt.plot(predictions_v, label="Learned Predictions", color="blue")
plt.plot(ideal_expectations, label="Ideal Expectations", color="orange")
plt.xlabel("Trials")
plt.ylabel("Values")
plt.title("Learned Predictions and Ideal Expectations")
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