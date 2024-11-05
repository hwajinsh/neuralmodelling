import numpy as np
import matplotlib.pyplot as plt
from rescorla_wagner import rescorla_wagner

## Explaining Away

# Parameters for the Rescorla-Wagner model
alpha = 0.1
num_trials_pretraining = 50 
num_trials_training = 50   
num_trials_result = 50   
total_trials = num_trials_pretraining + num_trials_training + num_trials_result 

stimuli_1 = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.ones(num_trials_result)])
stimuli_2 = np.concatenate([np.ones(num_trials_pretraining), np.zeros(num_trials_training), np.ones(num_trials_result)])

# Reward array (1 = reward, 0 = no reward in the Result Phase)
rewards = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.ones(num_trials_result)])

# Ideal expectations in blocking: reward is always present
ideal_expectations = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.ones(num_trials_result)])

# Apply Rescorla-Wagner rule
predictions_v, weights_1, weights_2 = rescorla_wagner(stimuli_1, stimuli_2, rewards, alpha)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(weights_1, label="Weight for Stimulus 1", color="blue")
plt.plot(weights_2, label="Weight for Stimulus 2", color="orange", linestyle="--")
plt.xlabel("Trials")
plt.ylabel("Weights")
plt.title("Expectations (Weights) for Stimulus 1 and Stimulus 2")
plt.legend()
plt.grid(True)
plt.show()

# Why does the Rescorla-Wagner rule failed to produce the correct expectations?
# Explaining away depends on a new quantity, namely the time within each trial, 
# Since a further positive association with the first stimulus is reliably established 
# Only if it precedes the second stimulus in the trials in which they are paired with a reward. 

# ???In other words, in this case, the Rescorla-Wagner rule incorrectly predicts inhibitory, not excitatory, secondary conditioning.
