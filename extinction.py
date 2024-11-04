import numpy as np
import matplotlib.pyplot as plt

## Extinction

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

# Step 2: Plot Expectation of Receiving US
# High expectation during conditioning, lower during extinction, moderate after delay
expectation = np.zeros(num_trials)

# Set expectations for each phase
expectation[:50] = 1.0  # High expectation in State 1
expectation[50:100] = 0.0  # Low expectation in State 2
expectation[100] = 0.5  # What do we expect? spontaneous full recovery? moderate? 

plt.figure(figsize=(10, 6))
plt.plot(expectation, label="Expectation of US")
plt.xlabel("Trial")
plt.ylabel("Expectation of US")
plt.title("Animal's Expectation of Receiving the US on Each Trial")
plt.legend()
plt.show()

# Step 3: Define Heuristic Function to Update Belief

## This needs major fixing lol

def update_belief(prev_state, similarity, time_gap):
    belief = np.zeros(3)
    decay_factor = 0.7  # Decay factor for transitioning belief

    if time_gap > 20:  # If large time gap, assume a shift to new context (state 3)
        belief[2] = 1.0  # Full belief in new state
    elif similarity < 0.5:  # If low similarity, partial shift to state 2
        belief[prev_state] = decay_factor
        belief[1] = 1 - decay_factor
    else:  # Otherwise, maintain belief in current state
        belief[prev_state] = 1.0

    return belief

# Step 4: Rescorla-Wagner Update for Association Strength
def update_association_strength(belief, prev_strength, learning_rate, reward):
    # Using Rescorla-Wagner rule: ΔV = α * (λ - V)
    prediction_error = reward - prev_strength
    return prev_strength + learning_rate * belief * prediction_error

# Initialize association strengths for each state
association_strengths = np.zeros((num_trials, 3))
learning_rate = 0.1

# Trial loop for updating association strengths
for trial in range(num_trials):
    reward = 1 if trial < 50 else 0  # Reward during conditioning phase only

    # For each state, update association strength weighted by belief
    for state in range(3):
        association_strengths[trial, state] = update_association_strength(
            belief_array[trial, state], association_strengths[trial - 1, state] if trial > 0 else 0,
            learning_rate, reward
        )

# Plot association strengths for each state
plt.figure(figsize=(10, 6))
for state in range(3):
    plt.plot(association_strengths[:, state], label=f"State {state+1}")
plt.xlabel("Trial")
plt.ylabel("Association Strength (CS-US)")
plt.title("Association Strength between CS and US for Each State")
plt.legend()
plt.show()