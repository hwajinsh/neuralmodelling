import numpy as np
import matplotlib.pyplot as plt

## Extinction Sample Work

# Step 1: Define Belief Array
# 50 trials for conditioning, 50 for extinction, and 1 post-delay
num_trials = 101
belief_array = np.zeros((num_trials, 2))  # 2 possible states

# During conditioning phase (trials 1-50), the animal is in State 1
belief_array[:50, 0] = 1.0  # 100% belief in State 1

# During extinction phase (trials 51-100), the animal is in State 2
belief_array[50:100, 1] = 1.0  # 100% belief in State 2

# After delay (trial 101), assume some belief in State 1 and 2 reemerges
belief_array[100, 0] = 0.5  # Partial return to State 1
belief_array[100, 1] = 0.5  # Partial belief in State 2

# Plot probabilities for each state
plt.figure(figsize=(10, 6))
plt.plot(belief_array[:, 0], label="Belief State 1", color="blue")
plt.plot(belief_array[:, 1], label="Belief State 2", color="orange")
plt.xlabel("Trial")
plt.ylabel("Probability of being in state")
plt.title("Probabilities of Being in State 1 and State 2 Across Trials")
plt.legend()
plt.show()

# Step 2: Define Heuristic Function to Update Belief

# Define parameters
belief_1 = np.zeros(num_trials)
belief_2 = np.zeros(num_trials)

# Time array: 1 for the first 100 trials, 30 for the last trial
time = np.ones(num_trials)
time[100] = 30  # 30-day delay before the 101st trial

# Set all stimuli to the same value
stimuli = np.ones(num_trials)  # All stimuli are the same (1)

# Initialize beliefs
belief_1[:50] = 1.0  # 100% belief in State 1 for trials 1-50
belief_2[50:100] = 1.0  # 100% belief in State 2 for trials 51-100
belief_1[100] = 0.5  # Partial belief in State 1 after the delay
belief_2[100] = 0.5  # Partial belief in State 2 after the delay

def update_belief(belief_1, similarity, time):
    # Calculate time-weighted factors
    time_weight = (1 / time) * 10  # Adjust time scale factor as needed

    # Calculate probabilities of state transition based on similarity
    prob_same = time_weight * similarity
    prob_diff = time_weight * (1 - similarity)

    # Update belief probabilities for state 1 and state 2
    prob_1 = belief_1 * prob_same + (1 - belief_1) * prob_diff
    prob_2 = belief_1 * prob_diff + (1 - belief_1) * prob_same

    # Normalize belief in State 1
    state_1 = prob_1 / (prob_1 + prob_2)
    return state_1

# Iterate over trials and update beliefs
for i in range(1, num_trials):
    # Since stimuli are the same, similarity will always be 1
    similarity = 1  # All stimuli are the same
 
    # Update belief for State 1 based on previous belief and similarity
    belief_1[i] = update_belief(belief_1[i - 1], similarity, time[i - 1])

    # Belief for State 2 is complementary to belief in State 1
    belief_2[i] = 1 - belief_1[i]

# Plot dynamic beliefs for each state across trials
plt.figure(figsize=(10, 6))
plt.plot(belief_1, label="Belief State 1 (Dynamic)", color="blue")
plt.plot(belief_2, label="Belief State 2 (Dynamic)", color="orange")
plt.xlabel("Trial")
plt.ylabel("Probability of being in state")
plt.title("Dynamic Probabilities of Being in State 1 and State 2 Across Trials")
plt.legend()
plt.xlim(0, num_trials - 1)
plt.xticks(np.arange(0, num_trials, step=10))  # Optional: Set x-ticks for clarity
plt.show()

# Step 3: Rescorla-Wagner Update for Association Strength
def extinction(belief_array, num_trials, alpha=0.1):
    weights_1 = np.zeros(num_trials)
    weights_2 = np.zeros(num_trials)
    predictions_v = np.zeros(num_trials)
    
    # Define punishment array (US is present during conditioning, absent during extinction)
    punishment = np.zeros(num_trials)
    punishment[:50] = 1  # Punishment during conditioning phase

    # Loop over each trial
    for i in range(1, num_trials):
        # Calculate predicted punishment based on belief states and previous weights
        predictions_v[i] = weights_1[i-1] * belief_array[i-1, 0] + weights_2[i-1] * belief_array[i-1, 1]
        
        # Calculate prediction error
        error = punishment[i] - predictions_v[i]
        
        # Update weights based on prediction error and belief states
        weights_1[i] = weights_1[i-1] + alpha * error * belief_array[i-1, 0]
        weights_2[i] = weights_2[i-1] + alpha * error * belief_array[i-1, 1]

    return weights_1, weights_2

# Run the Rescorla-Wagner extinction process
weights_1, weights_2 = extinction(belief_array, num_trials)

# Plot the weights for each state over trials
plt.figure(figsize=(10, 6))
plt.plot(weights_1, label="Expectation of punishment in belief state 1", color="blue")
plt.plot(weights_2, label="Expectation of punishment in belief state 2", color="orange", linestyle="--")
plt.xlabel("Trials")
plt.ylabel("Expectation")
plt.title("Relationship between belief states and punishment expectations")
plt.legend()
plt.show()