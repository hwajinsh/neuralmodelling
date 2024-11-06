import numpy as np
import matplotlib.pyplot as plt
from rescorla_wagner import rescorla_wagner

## Inhibitory Conditioning

## We need more trials in training and testing (specially in training) so that it
# does not fluctuate in the beginning and it stabilizes

# Parameters for the Rescorla-Wagner model
epsilon = 0.1  # Learning rate for stimuli (CS1, CS2)
num_trials_training = 200     # Trials in Training (CS1 + CS2)
#num_trials_result = 200     # Trials in Result phase for visualizing expectations
total_trials = num_trials_training 

stimuli_1 = np.concatenate([np.ones(num_trials_training)])
stimuli_2 = np.concatenate([
    np.tile([1, 0], num_trials_training // 2) # CS2 is present only when CS1 is not rewarded
])

#stimuli_2 = np.ones(total_trials)
#stimuli_2[1::2] = 0 

# Reward array (difference between stimuli_1 and stimuli_2 or inhibitory stimuli)
rewards = stimuli_1 - stimuli_2

# Ideal expectations in binhibitory conditioning: (difference between stimuli_1 and stimuli_2 or inhibitory stimuli)
ideal_expectations = stimuli_1 - stimuli_2

# Apply Rescorla-Wagner rule
predictions_v, weights_1, weights_2 = rescorla_wagner(stimuli_1, stimuli_2, rewards, epsilon)

# Plot expectations (Weights) for both stimuli
plt.figure(figsize=(10, 5))
plt.plot(weights_1, label="Weight for Stimulus 1", color="blue")
plt.plot(weights_2, label="Weight for Stimulus 2", color="orange")
plt.xlabel("Trials")
plt.ylabel("Weights")
plt.title("Expectations (Weights) for Stimulus 1 and Stimulus 2")
plt.legend()
plt.grid(True)
plt.show()