import numpy as np
import matplotlib.pyplot as plt

# Step 1: Initialize Belief Arrays
num_trials = 101
belief_1 = np.zeros(num_trials)
belief_2 = np.zeros(num_trials)

# Set initial conditions
belief_1[0] = 1.0  # Start with 100% belief in State 1
belief_2[0] = 0.0

# Define the time intervals (1 for each of the first 100 trials, 31 for the 101st)
time = np.ones(num_trials)
time[100] = 31  # 1-day trial + 30-day delay before for the last trial

# Define stimuli, which are all the same, so state_similarity will always be 1
stimuli = np.concatenate([np.ones(50), np.zeros(50), np.ones(1)])

# Define the heuristic function with additional debug output
def state_beliefs_heuristic(belief_state_1, state_similarity, time):
    # Increase the time_weight factor to 50 to amplify its effect
    time_weight = (1 / time) * 50
    prob_same_state = time_weight * state_similarity
    prob_diff_state = time_weight * (1 - state_similarity)
    
    prob_state_1 = belief_state_1 * prob_same_state + (1 - belief_state_1) * prob_diff_state
    prob_state_2 = belief_state_1 * prob_diff_state + (1 - belief_state_1) * prob_same_state
    
    # Normalize to calculate the probability of being in State 1
    state_1 = prob_state_1 / (prob_state_1 + prob_state_2)
    
    # Debugging output to check calculations
    print(f"time: {time}, time_weight: {time_weight}, prob_same_state: {prob_same_state}, prob_diff_state: {prob_diff_state}, state_1: {state_1}")
    
    return state_1

# Iterate over trials to update beliefs
for i in range(1, num_trials):
    # Check similarity with previous stimulus
    ## This is wrong
    state_similarity = 1 if stimuli[i] == stimuli[i - 1] else 0
    
    # Update belief for State 1 using the heuristic function
    belief_1[i] = state_beliefs_heuristic(belief_1[i - 1], state_similarity, time[i - 1])
    
    # Belief in State 2 is complementary to belief in State 1
    belief_2[i] = 1 - belief_1[i]

    # Debugging output to see belief values at each trial
    print(f"Trial {i}: belief_1 = {belief_1[i]}, belief_2 = {belief_2[i]}")

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