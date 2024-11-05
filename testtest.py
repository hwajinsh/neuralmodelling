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

# Initialize New Belief Arrays
num_trials = 101
belief_1 = np.zeros(num_trials)
belief_2 = np.zeros(num_trials)

# Set initial conditions
belief_1[0] = 1.0  # Start with 100% belief in State 1
belief_2[0] = 0.0

# Define the time intervals (1 for each of the first 100 trials, 31 for the 101st)
time = np.ones(num_trials)
time[100] = 31  # 1-day trial + 30-day delay before for the last trial

# Define the stimuli (1 for present CS; 0 for )
stimuli = np.concatenate([np.ones(50), np.zeros(50), np.ones(1)])

# Define the heuristic function with additional debug output
def state_beliefs_heuristic(belief_1, similarity, time):
    time_weight = (1 / time)
    prob_same_state = time_weight * similarity
    prob_diff_state = time_weight * (1 - similarity)
    
    prob_state_1 = belief_1 * prob_same_state + (1 - belief_1) * prob_diff_state
    prob_state_2 = belief_1 * prob_diff_state + (1 - belief_1) * prob_same_state
    
    # Normalize to calculate the probability of being in State 1
    state_1 = prob_state_1 / (prob_state_1 + prob_state_2)
    
    # Debugging output to check calculations
    print(f"time: {time}, time_weight: {time_weight}, prob_same_state: {prob_same_state}, prob_diff_state: {prob_diff_state}, state_1: {state_1}")
    
    return state_1

# Iterate over trials to update beliefs
for i in range(1, num_trials):
    # Check similarity with previous stimulus
    state_similarity = 1 if stimuli[i] == stimuli[i - 1] else 0
    
    # Update belief for State 1 using the heuristic function
    belief_1[i] = state_beliefs_heuristic(belief_1[i - 1], state_similarity, time[i])
    
    # Belief in State 2 is complementary to belief in State 1
    belief_2[i] = 1 - belief_1[i]

    # Debugging output to see belief values at each trial
    print(f"Trial {i}: belief_1 = {belief_1[i]}, belief_2 = {belief_2[i]}")

# Plot dynamic beliefs for each state across trials
plt.figure(figsize=(10, 6))
plt.plot(belief_1, label="Belief State 1", color="blue")
plt.plot(belief_2, label="Belief State 2", color="orange")
plt.xlabel("Trial")
plt.ylabel("Probability of being in state")
plt.title("Heuristic Probabilities of Being in State 1 and State 2 Across Trials")
plt.legend()
plt.xlim(0, num_trials - 1)
plt.xticks(np.arange(0, num_trials, step=10))  # Optional: Set x-ticks for clarity
plt.show()

# I got why this goes up to 1; in the exercise sheet it says:
# for this part of the exercise, ignore the learning of the CS-US relationship, 
# assume that the animal perfectly knows the connection between CS and US for each state.
# so it makes sense here that animal perfectly knows the connection between CS and US even after the extinction training.

# Step 3: Rescorla-Wagner Update for Association Strength

def extinction(belief_1, belief_2, num_trials, learning_rate = 0.1):
    weights_1 = np.zeros(num_trials)
    weights_2 = np.zeros(num_trials)
    predictions = np.zeros(num_trials)
    
    # Define punishment array (US is present during conditioning, absent during extinction)
    punishment = np.zeros(num_trials)
    punishment[:50] = 1  # Punishment during conditioning phase

    # Loop over each trial
    for i in range(1, num_trials):
        # Calculate predicted punishment based on belief states and previous weights
        predictions[i] = weights_1[i-1] * belief_1[i-1] + weights_2[i-1] * belief_2[i-1]
        
        # Calculate prediction error
        error = punishment[i] - predictions[i]
        
        # Update weights based on prediction error and belief states
        weights_1[i] = weights_1[i-1] + learning_rate * error * belief_1[i-1]
        weights_2[i] = weights_2[i-1] + learning_rate * error * belief_2[i-1]

    return weights_1, weights_2

# Run the Rescorla-Wagner extinction process
weights_1, weights_2 = extinction(belief_1, belief_2, num_trials)

# Plot the weights for each state over trials
plt.figure(figsize=(10, 6))
plt.plot(weights_1, label="Expectation of punishment in belief state 1", color="blue")
plt.plot(weights_2, label="Expectation of punishment in belief state 2", color="orange", linestyle="--")
plt.xlabel("Trials")
plt.ylabel("Expectation")
plt.title("Relationship Between Belief States and Punishment Expectations")
plt.legend()
plt.show()

# And according to RW-rule, there is no reason for state 2 to be associated with a punishment.