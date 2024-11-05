import numpy as np
import matplotlib.pyplot as plt

## Secondary Conditioning

# Parameters for the Rescorla-Wagner model
learning_rate = 0.15  # Learning rate for stimuli (CS1, CS2)
num_trials_pretraining = 50  # Trials in Pre-Training (CS1 only)
num_trials_training = 50     # Trials in Training (CS1 + CS2)
num_trials_result = 50       # Trials in Result phase for visualizing expectations
total_trials = num_trials_pretraining + num_trials_training + num_trials_result

# Stimuli presentation arrays for the three stages
stimuli_CS1 = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.zeros(num_trials_result)])
stimuli_CS2 = np.concatenate([np.zeros(num_trials_pretraining), np.ones(num_trials_training), np.ones(num_trials_result)])

# Reward array (1 = reward, 0 = no reward in the Result Phase)
rewards = np.concatenate([np.ones(num_trials_pretraining), np.zeros(num_trials_training), np.ones(num_trials_result)])

# Arrays to store expectations for CS1 and CS2 over trials
expectations_CS1 = np.zeros(total_trials)
expectations_CS2 = np.zeros(total_trials)

# Initial expectations
V_CS1 = 0.0
V_CS2 = 0.0

for t in range(total_trials):
    # Store the current expectations
    expectations_CS1[t] = V_CS1
    expectations_CS2[t] = V_CS2
    
    # Total expectation
    total_expectation = V_CS1 * stimuli_CS1[t] + V_CS2 * stimuli_CS2[t]
    
    # Calculate prediction error (for the current trial)
    prediction_error = rewards[t] - total_expectation
    
    # Update expectations based on which CS is present
    # Update only CS1; CS2 should only be updated if it had a chance of predicting the reward.
    V_CS1 += learning_rate * prediction_error * stimuli_CS1[t]
    V_CS2 += learning_rate * prediction_error * stimuli_CS2[t]

# Plotting the three stages with blocking effect
plt.figure(figsize=(12, 6))
plt.plot(expectations_CS1, label="Expectation for CS1")
plt.plot(expectations_CS2, label="Expectation for CS2", linestyle="--")
plt.axvline(x=num_trials_pretraining, color="grey", linestyle=":", label="Start of Training Phase")
plt.axvline(x=num_trials_pretraining + num_trials_training, color="grey", linestyle=":", label="Start of Result Phase")
plt.xlabel("Trials")
plt.ylabel("Expectation of Reward")
plt.title("Secondary Paradigm with Pre-Training, Training, and Result Phases")
plt.legend()
plt.show()

# Why does the Rescorla-Wagner rule failed to produce the correct expectations?
# Since the reward does not appear when the second stimulus is presented, the delta rule would cause w2 to become negative. 
# In other words, in this case, the Rescorla-Wagner rule incorrectly predicts inhibitory, not excitatory, secondary conditioning.