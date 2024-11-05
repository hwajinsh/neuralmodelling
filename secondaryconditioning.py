import numpy as np
import matplotlib.pyplot as plt
from rescorla_wagner import rescorla_wagner

## Secondary Conditioning

# Parameters for the Rescorla-Wagner model
alpha = 0.1  # Learning rate for stimuli (CS1, CS2)
num_trials_pretraining = 50  # Trials in Pre-Training (CS1 only)
num_trials_training = 50     # Trials in Training (CS1 + CS2)
num_trials_result = 50       # Trials in Result phase for visualizing expectations
total_trials = num_trials_pretraining + num_trials_training + num_trials_result

num_trials=1000
# Stimuli presentation arrays for the three stages
stimuli_1 = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.zeros(num_trials_result)])
stimuli_2 = np.concatenate([np.zeros(num_trials_pretraining), np.ones(num_trials_training), np.ones(num_trials_result)])

# Reward array (1 = reward, 0 = no reward in the Result Phase)
#rewards = np.concatenate([np.ones(num_trials_pretraining), np.zeros(num_trials_training), np.ones(num_trials_result)])

# Ideal expectations in blocking: reward is always present
#ideal_expectations = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.ones(num_trials_result)])

#Lotta's reasoning

primary_s = np.ones(num_trials)  # Primary stimulus always present
secondary_s = np.concatenate([np.zeros(num_trials // 2), np.ones(num_trials // 2)])  # Secondary stimulus in second half
rewards = np.concatenate([np.ones(num_trials // 2), np.zeros(num_trials // 2)])  # Rewards for first half only
ideal_expectations = np.ones(num_trials)
# Apply Rescorla-Wagner rule
predictions_v, weights_1, weights_2 = rescorla_wagner(primary_s, secondary_s, rewards, alpha)

# Plot: Learned predictions, ideal expectations, and stimulus 2
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
plt.plot(weights_1, label="Weight for Stimulus 1", color="purple")
plt.plot(weights_2, label="Weight for Stimulus 2", color="orange")
plt.xlabel("Trials")
plt.ylabel("Weights")
plt.title("Expectations (Weights) for Stimulus 1 and Stimulus 2")
plt.legend()
plt.grid(True)
plt.show()
# Why does the Rescorla-Wagner rule failed to produce the correct expectations?
# Since the reward does not appear when the second stimulus is presented, the delta rule would cause w2 to become negative. 
# In other words, in this case, the Rescorla-Wagner rule incorrectly predicts inhibitory, not excitatory, secondary conditioning.