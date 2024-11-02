import numpy as np
import matplotlib.pyplot as plt

## Explaining Away

# Parameters for the Rescorla-Wagner model
learning_rate = 0.1
num_trials_pretraining = 50 
num_trials_training = 50     
num_trials_result = 50       
total_trials = num_trials_pretraining + num_trials_training + num_trials_result 

stimuli_CS1 = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.ones(num_trials_result)])
stimuli_CS2 = np.concatenate([np.ones(num_trials_pretraining), np.zeros(num_trials_training), np.ones(num_trials_result)])

# Reward array (1 = reward, 0 = no reward in the Result Phase)
rewards_CS1 = np.concatenate([np.ones(num_trials_pretraining), np.ones(num_trials_training), np.ones(num_trials_result)])
rewards_CS2 = np.concatenate([np.ones(num_trials_pretraining), np.zeros(num_trials_training), np.zeros(num_trials_result)])

# Arrays to store expectations for CS1 and CS2 over trials
expectations_CS1 = np.zeros(total_trials)
expectations_CS2 = np.zeros(total_trials)

# Initial expectations
V_CS1 = 0.0
V_CS2 = 0.0

# Apply the Rescorla-Wagner rule over trials
for t in range(total_trials):
    # Store the current expectations
    expectations_CS1[t] = V_CS1
    expectations_CS2[t] = V_CS2
    
    # Calculate total expectation
    total_expectation = V_CS1 + V_CS2
    
    # Calculate prediction error
    prediction_error_CS1 = rewards_CS1[t] - total_expectation
    prediction_error_CS2 = rewards_CS2[t] - total_expectation
    
    # Update expectations based on which CS is present
    V_CS1 += learning_rate * prediction_error_CS1 * stimuli_CS1[t]

    if t < num_trials_pretraining or t >= num_trials_pretraining + num_trials_training:
        V_CS2 += learning_rate * prediction_error_CS2 * stimuli_CS2[t]

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(expectations_CS1, label="Expectation for CS1")
plt.plot(expectations_CS2, label="Expectation for CS2", linestyle="--")
plt.axvline(x=num_trials_training, color="grey", linestyle=":", label="Start of Result Phase")
plt.axvline(x=num_trials_pretraining + num_trials_training, color="grey", linestyle=":", label="Start of Result Phase")
plt.xlabel("Trials")
plt.ylabel("Expectation of Reward")
plt.title("Secondary Conditioning with CS1 and CS2")
plt.legend()
plt.grid()
plt.show()
